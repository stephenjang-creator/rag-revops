# Observability & Monitoring

Deal Desk Helper is instrumented end-to-end across three phases: request tracing
(Phase 1), operational metrics (Phase 2), and regression gating in CI (Phase 3).
Tracing to Langfuse is **env-gated and off by default**, so the deployed
bring-your-own-key demo runs clean; the local metrics that feed the CI gate work
with **zero hosted infrastructure**.

## Design principles

1. **Off by default, one flip to enable.** Langfuse tracing activates only when
   `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are in the environment. Absent →
   every decorator and helper is a transparent no-op (the `@observe` decorator
   returns the *original* function object, so there isn't even an extra stack
   frame). Proven by test in both directions.
2. **Fail-safe.** An observability error never propagates into the pipeline. A
   monitoring bug cannot break the thing it monitors.
3. **Metrics without infra.** Operational signals come from a local JSONL sink
   that records every request regardless of Langfuse, so Phase 2 and the Phase 3
   gate work with no hosted service.

## Phase 1 — Request tracing

Every stage of the pipeline is a traced observation. For any single request you
can see, in Langfuse:

| Signal | Where it's captured |
|---|---|
| Which chunks were retrieved (ids, docs, scores, order) | `graph.py::_retrieve` → `record(retrieved=[...])` |
| How the reranker reordered them (in/out counts, dropped, score range, new order) | `rerank.py::rerank` → `record(reordered_ids=[...], dropped=...)` |
| The exact prompt sent to the model (system + user) | `generation.py::generate` → `record(system_prompt=..., user_prompt=...)` |
| The model's response | `generation.py::generate` → `record(answer=...)` |
| Tokens consumed (and cost, computed by Langfuse) | every LLM call → `record_usage(model, input_tokens, output_tokens)` |
| Query rewrite (before/after, tokens) | `query_rewrite.py::rewrite` |
| Judge membership decisions (batch size, members/rejected) | `analytical_generation.py::_judge_batch` |

The four model touchpoints — query rewrite, generation, membership judge, and
criterion guidance — are marked `as_type="generation"` so Langfuse treats their
token/cost fields specially. Every request opens a named root trace
(`single_doc_query`, `search_all_query`, `analytical_query`, `eval_query`) tagged
with the config version that produced it.

## Phase 2 — Operational metrics (SRE view)

`RequestTimer` (in `observability.py`) wraps each query entry point and writes one
JSON line per request to the sink (`RAG_METRICS_PATH`, default
`metrics/requests.jsonl`). `rag_revops.metrics_report` reads that sink and computes
the signals averages hide:

- **Latency p50 / p95 / p99 / max** — nearest-rank percentiles, not just the mean.
  The mean is printed *alongside* p95 specifically to make the tail visible.
- **LLM calls per request** — p50 and max, plus total. Cross-corpus queries fan
  out to many judge calls; this is where that shows up.
- **Citation coverage** — the share of *answered* requests (declines and errors
  excluded from the denominator) that carry ≥1 inline citation. This is the
  "properly grounded" metric.
- **Failure rate** — the share of requests that raised. `RequestTimer` records the
  error and re-raises (never swallows), so failures are both surfaced to the user
  and counted.
- **Decline rate** — reported separately from failures: a decline is correct
  behavior, not an error.

```bash
# human-readable report over the whole sink
python -m rag_revops.metrics_report          # or: rag-metrics

# last hour only, as JSON
python -m rag_revops.metrics_report --since-min 60 --json
```

## Phase 3 — Regression gating

Two gates run in CI (`.github/workflows/ci.yml`, `faithfulness-gate` job):

1. **Faithfulness gate** (pre-existing) — `eval/run_eval.py --gate` fails the
   build if mean faithfulness drops below `eval.min_faithfulness` (0.70, a
   regression gate calibrated below the measured ~0.78 baseline).
2. **Operational-metrics gate** (new) — because the eval ran every golden
   question through the live pipeline, `metrics/requests.jsonl` now holds real
   per-request data. `metrics_report --gate` fails the build if citation coverage,
   failure rate, or p95 latency breach their thresholds. No extra API spend — it
   reads traffic that already happened.

```yaml
python -m rag_revops.metrics_report --gate \
  --min-coverage 0.90 --max-failure-rate 0.05 --max-p95-ms 90000
```

A threshold whose metric is unmeasurable (no traffic) warns and is skipped rather
than failing on absence of data.

**Prompt & config versioning** (Phase 3.3) was already in place: prompts live in
`config/settings.yaml` (versioned config, not string literals), and `settings.version`
is stamped onto every trace and metrics record. In CI the record's `config_version`
is set to the commit SHA, so a prompt change is directly attributable to any metric
shift it causes — a prompt edit is treated as a code change.

## Enabling Langfuse

```bash
pip install -e ".[observability]"     # installs langfuse

# .env (gitignored)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or self-hosted
```

Nothing else changes. Remove the keys and the system is inert again.

## Deployment behavior (Render)

The two subsystems are controlled independently, which is what makes the hosted
demo safe:

| Subsystem | Deployed default | How to control |
|---|---|---|
| **Langfuse tracing** | **Off** | Only turns on if `LANGFUSE_*` env vars are set on the Render service. Leave them unset → visitor queries never reach your Langfuse project. |
| **Local metrics sink** | **On (silent)** | Collects one JSONL line per request so you get real traffic. Set `RAG_METRICS_PATH=""` to disable, or point it at a persistent disk to keep history across restarts. |
| **Admin metrics endpoint** | **Locked** | `GET /api/metrics` returns the stats only when the request carries an `X-Admin-Token` header matching the `ADMIN_TOKEN` env var. No `ADMIN_TOKEN` set → the endpoint stays locked. |

So on the deployed app: keep `LANGFUSE_*` off the Render env (tracing stays off),
let the sink collect quietly, and set `ADMIN_TOKEN` so only you can read
`/api/metrics`.

### Admin views

Both read the same local JSONL sink (so they work without Langfuse) and surface
live p50/p95/p99 latency, citation coverage, failure/decline rates, and
LLM-calls-per-request:

* **Hosted (Render):** `GET /api/metrics` with an `X-Admin-Token` header. Configure
  it as a Render environment variable:

  ```
  ADMIN_TOKEN=something-only-you-know
  ```

* **Local (Streamlit):** `rag_revops.admin_panel.render_admin_panel()` renders the
  same stats behind a password box in the `app.py` sidebar — it reads
  `st.secrets["admin_password"]` (put it in `.streamlit/secrets.toml`, gitignored)
  and renders nothing if no password is configured.

Note: the demo image bakes the index in, but the container filesystem is
ephemeral, so the sink reflects traffic since the last restart — a live-session
view, not long-term history. For durable history, enable Langfuse (persists
server-side) or point `RAG_METRICS_PATH` at a persistent disk.

## Files

- `src/rag_revops/observability.py` — the layer: `@observe`, `record`,
  `record_usage`, `score`, `trace_context`, `flush`, `RequestTimer`, `count_llm_call`.
- `src/rag_revops/metrics_report.py` — the Phase 2/3 reporter + gate (`rag-metrics`).
- `tests/test_observability.py` — no-op-path and metrics-math tests (keyless).
- Instrumented in place: `graph.py`, `generation.py`, `rerank.py`,
  `query_rewrite.py`, `analytical_generation.py`.
