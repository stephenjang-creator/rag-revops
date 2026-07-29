"""Event schema shared by the recorded demo (`/api/demo`) and the live stream
(`/api/ask`). Each event is ``{"type": <name>, "data": {...}}``. The frontend
renders both sources with one handler; the only difference is transport — the
demo replays events on a timer, the live path streams them over SSE.

`label`/`detail`/`ms` on a stage event are exactly what a trace row renders, so
the client needs no lookup table.
"""

from __future__ import annotations

import json
from typing import Any


def _ev(name: str, **data: Any) -> dict[str, Any]:
    return {"type": name, "data": data}


def stage(label: str, detail: str, ms: float | str, state: str = "done") -> dict[str, Any]:
    return _ev("stage", label=label, detail=detail, ms=ms, state=state)


def route(skill: str, label: str, reason: str, needs_clarification: bool = False) -> dict:
    return _ev("route", skill=skill, label=label, reason=reason,
               needs_clarification=needs_clarification)


def rewrite(original: str, rewritten: str, changed: bool) -> dict[str, Any]:
    return _ev("rewrite", original=original, rewritten=rewritten, changed=changed)


def retrieval(n_candidates: int, n_reranked: int, top_score: float | None) -> dict:
    return _ev("retrieval", n_candidates=n_candidates, n_reranked=n_reranked,
               top_score=top_score)


def judge(doc_id: str, verdict: bool, tag: str, reason: str) -> dict[str, Any]:
    return _ev("judge", doc_id=doc_id, verdict=verdict, tag=tag, reason=reason)


def answer(kind: str, **fields: Any) -> dict[str, Any]:
    """kind ∈ {draft, single, find}. fields carry the render payload — heading,
    subheading, body_html, body_plain, note, disclaimer, sources, findings,
    summary — whatever that result type needs."""
    return _ev("answer", kind=kind, **fields)


def decline(reason: str, why: str) -> dict[str, Any]:
    return _ev("decline", reason=reason, why=why)


def error(code: str, message: str) -> dict[str, Any]:
    return _ev("error", code=code, message=message)


def done(total_ms: float | None = None, llm_calls: int | None = None) -> dict[str, Any]:
    return _ev("done", total_ms=total_ms, llm_calls=llm_calls)


def sse(event: dict[str, Any]) -> str:
    """Serialize one event as an SSE frame: a named event + JSON data."""
    return f"event: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


def heartbeat() -> str:
    """An SSE comment line — keeps the connection (and the client spinner) alive
    during the long analytical judge fan-out without emitting a real event."""
    return ": keep-alive\n\n"
