"""FastAPI service: streams the pipeline as SSE and serves the static page.

Endpoints
  POST /api/ask        — SSE; one event per trace row (the one that matters)
  GET  /api/demo       — the recorded runs, same event schema (keeps the page keyless)
  GET  /api/contracts  — doc ids with readable titles for the single-contract picker
  GET  /api/health     — corpus/index status
  GET  /api/metrics    — operational metrics (admin-token gated)
  GET  /                — the static case-study page

Keys are bring-your-own, per request, in headers — never logged, never stored.
Concurrency is capped to one live run at a time (the judge fan-out is ~11 LLM
calls); a second concurrent live request gets a `busy` event and the client falls
back to the recorded run.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import events as E
from .contracts import list_contracts
from .demo_data import demo_runs
from .runner import run_live

_STATIC = Path(__file__).resolve().parent / "static"
_LIVE_MODES = {"auto", "draft", "find", "single"}
_SENTINEL = object()

app = FastAPI(title="Deal Desk Helper", docs_url=None, redoc_url=None)

# One live run at a time — the analytical judge fan-out will rate-limit a trial
# key if two visitors run simultaneously.
_live_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# /api/ask — the SSE endpoint
# ---------------------------------------------------------------------------
def _swap_keys(anthropic_key: str, cohere_key: str) -> tuple[str | None, str | None]:
    from rag_revops.config import load_secrets

    prev = (os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("COHERE_API_KEY"))
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["COHERE_API_KEY"] = cohere_key
    load_secrets.cache_clear()  # so the pipeline reads these keys, not cached ones
    return prev


def _restore_keys(prev: tuple[str | None, str | None]) -> None:
    from rag_revops.config import load_secrets

    for name, val in zip(("ANTHROPIC_API_KEY", "COHERE_API_KEY"), prev, strict=True):
        if val is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = val
    load_secrets.cache_clear()


async def _ask_stream(question: str, mode: str, doc_id: str | None,
                      anthropic_key: str, cohere_key: str):
    if mode not in _LIVE_MODES:
        yield E.sse(E.error("bad_request", f"unknown mode '{mode}'"))
        return
    if not (anthropic_key and cohere_key):
        yield E.sse(E.error("keys_required",
                            "Paste your Anthropic and Cohere keys to run a live question."))
        return
    if _live_lock.locked():
        msg = "A live run is already in progress — showing the recorded run instead."
        yield E.sse(E.error("busy", msg))
        return

    async with _live_lock:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def emit(ev: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        def work() -> None:
            try:
                run_live(question, mode, doc_id, emit)
            except Exception as exc:  # noqa: BLE001 - surface as an inline error event
                emit(E.error("internal", str(exc)[:300]))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        prev = _swap_keys(anthropic_key, cohere_key)
        fut = loop.run_in_executor(None, work)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield E.heartbeat()  # keep the SSE connection + spinner alive
                    continue
                if ev is _SENTINEL:
                    break
                yield E.sse(ev)
        finally:
            await fut
            _restore_keys(prev)


@app.post("/api/ask")
async def ask(
    request: Request,
    x_anthropic_key: str = Header(default=""),
    x_cohere_key: str = Header(default=""),
):
    body = await request.json()
    question = (body.get("question") or "").strip()
    mode = body.get("mode") or "auto"
    doc_id = body.get("doc_id")
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    stream = _ask_stream(question, mode, doc_id, x_anthropic_key, x_cohere_key)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Static/data endpoints
# ---------------------------------------------------------------------------
@app.get("/api/demo")
async def demo() -> dict[str, Any]:
    return {"runs": demo_runs()}


@app.get("/api/contracts")
async def contracts() -> dict[str, Any]:
    try:
        from rag_revops.config import load_settings
        from rag_revops.vectorstore import ChromaStore

        store = ChromaStore(load_settings().vectorstore)
        return {"contracts": list_contracts(store.list_contracts())}
    except Exception as exc:  # noqa: BLE001
        return {"contracts": [], "error": str(exc)[:200]}


@app.get("/api/health")
async def health() -> dict[str, Any]:
    from rag_revops.config import load_settings

    settings = load_settings()
    index_ready, corpus_size = False, 0
    try:
        from rag_revops.vectorstore import ChromaStore

        corpus_size = ChromaStore(settings.vectorstore).count()
        index_ready = corpus_size > 0
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok" if index_ready else "degraded",
        "corpus_size": corpus_size,
        "index_ready": index_ready,
        "config_version": settings.version,
    }


@app.get("/api/metrics")
async def metrics(x_admin_token: str = Header(default="")) -> JSONResponse:
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from rag_revops.metrics_report import compute, load_records
        from rag_revops.observability import metrics_sink_path

        path = metrics_sink_path()
        if path is None or not path.exists():
            return JSONResponse({"available": False})
        return JSONResponse({"available": True, "metrics": compute(load_records(path, None))})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"available": False, "error": str(exc)[:200]})


# Static page last, as the catch-all (so /api/* wins).
if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
