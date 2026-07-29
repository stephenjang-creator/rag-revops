# Deal Desk Helper — web service

A FastAPI app that streams the `rag_revops` pipeline as Server-Sent Events and
serves the static case-study page. It's the Render replacement for the Streamlit
UI. **One event schema, two sources:** the recorded runs (`/api/demo`) and the
live stream (`/api/ask`) render through the same trace panel, so a visitor with
no API keys still watches the pipeline work.

## Endpoints

| Route | What it does |
|---|---|
| `POST /api/ask` | SSE. One event per trace row: `stage` / `route` / `rewrite` / `retrieval` / `judge` / `answer` / `decline` / `error` / `done`. Keys are per-request headers (`X-Anthropic-Key`, `X-Cohere-Key`); one live run at a time. |
| `GET /api/demo` | The recorded runs as event streams — replayed client-side so the page is keyless. |
| `GET /api/contracts` | Doc ids with readable titles/years for the single-contract picker. |
| `GET /api/health` | `{status, corpus_size, index_ready, config_version}`. |
| `GET /api/metrics` | Operational metrics from `metrics_report` (gated by `X-Admin-Token` == `ADMIN_TOKEN`). |
| `GET /` | The static page (`web/static/`). |

## Run locally

```bash
pip install -e ".[web]"
python -m uvicorn web.server:app --reload --port 8000
# open http://localhost:8000  — the recorded runs work with no keys;
# paste Anthropic + Cohere keys under "Use your own API keys" to stream live.
```

The pipeline reads the committed demo index at `data/processed/chroma`, so the
keyless demo and `/api/contracts` work out of the box.

## Deploy (Render)

`render.yaml` defines one Docker web service. The `Dockerfile` bakes the committed
demo index into the image (no re-ingest on cold start) and runs
`uvicorn web.server:app`. BYO-key: no API keys in the service env; visitors bring
their own per request. Health check is `/api/health`.

## Key handling (BYO-key)

Keys arrive per request in headers, are held only for that request (set into the
environment under a single-live-run lock, then restored), and are never logged or
written to disk. A second concurrent live request gets a `busy` event and the
client falls back to the recorded run.
