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

## Deployment behavior (Streamlit Community Cloud)

The two subsystems are controlled independently, which is what makes the deployed
demo safe:

| Subsystem | Deployed default | How to control |
|---|---|---|
| **Langfuse tracing** | **Off** | Only turns on if `LANGFUSE_*` secrets are set in Streamlit. Leave them unset → reviewer queries never reach your Langfuse project. |
| **Local metrics sink** | **On (silent)** | Collects one JSONL line per request so you get real production traffic. Set `RAG_METRICS_PATH=""` to disable entirely. |
| **Admin metrics panel** | **Hidden** | Renders only for someone who enters `admin_password` (from `st.secrets`). No password configured → panel is invisible and unannounced. |

So on the deployed app: keep `LANGFUSE_*` out of Streamlit secrets (tracing stays
off), let the sink collect quietly, and set `admin_password` in Streamlit secrets
so only you can open the live metrics view.

### Admin panel

`rag_revops.admin_panel.render_admin_panel()` is called from the app sidebar. It:

* reads `st.secrets["admin_password"]`; if absent, renders nothing (metrics still
  collect);
* shows a password box under a "🔒 Admin" expander;
* on the correct password, renders live p50/p95/p99 latency, citation coverage,
  failure/decline rates, and LLM-calls-per-request from the same sink the CLI
  reads — with an optional time window (all-time / last 60 min / last 24 h).

Configure it in Streamlit Cloud → Settings → Secrets:

```toml
admin_password = "something-only-you-know"
```

Locally, put the same line in `.streamlit/secrets.toml` (gitignored). The metrics
sink and the admin panel both work without Langfuse, since they read the local
JSONL — so the deployed app has a live operational view even with tracing off.

Note: Streamlit Cloud's filesystem is ephemeral, so the deployed sink reflects
traffic since the last app restart — a live-session view, not long-term history.
For durable history, enable Langfuse (which persists server-side) or point
`RAG_METRICS_PATH` at persistent storage.

## Files

- `src/rag_revops/observability.py` — the layer: `@observe`, `record`,
  `record_usage`, `score`, `trace_context`, `flush`, `RequestTimer`, `count_llm_call`.
- `src/rag_revops/metrics_report.py` — the Phase 2/3 reporter + gate (`rag-metrics`).
- `tests/test_observability.py` — no-op-path and metrics-math tests (keyless).
- Instrumented in place: `graph.py`, `generation.py`, `rerank.py`,
  `query_rewrite.py`, `analytical_generation.py`.
