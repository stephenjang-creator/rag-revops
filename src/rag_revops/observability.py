"""Observability for Deal Desk Helper — tracing (Phase 1) + metrics (Phase 2).

Design contract
---------------
* **Env-gated, off by default.** Tracing to Langfuse activates only when
  ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` are present. Absent → every
  function here is a transparent no-op. The pipeline imports the same names
  either way; nothing branches at the call site.
* **Fail-safe.** An observability error never propagates into the pipeline —
  monitoring must not break the thing it monitors.
* **Local metrics always work.** SRE metrics (latency percentiles, calls per
  request, citation coverage, failure rate) are computed from a local JSONL
  sink that records every request regardless of whether Langfuse is on. That
  keeps Phase 2 functional with zero hosted infra, and gives the Phase 3 CI
  gate a file to read.

Enable Langfuse:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://us.cloud.langfuse.com   # optional; defaults to US region

Control the local metrics sink (independent of Langfuse):
    RAG_METRICS_PATH=metrics/requests.jsonl    # default; set empty to disable
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

logger = logging.getLogger("rag_revops.observability")

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Langfuse enablement — decided once at import time.
# ---------------------------------------------------------------------------

def _tracing_requested() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


_ENABLED = False
_langfuse = None
_observe_impl: Callable | None = None

# Langfuse data region host. Hardcoded to the US region — the keys and host must
# match the region your project lives in (regions are fully isolated). Still
# overridable via LANGFUSE_HOST if you ever move regions or self-host.
_DEFAULT_LANGFUSE_HOST = "https://us.cloud.langfuse.com"

if _tracing_requested():
    try:
        from langfuse import Langfuse
        from langfuse import observe as _lf_observe

        _langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            # `or` (not get's default): a present-but-empty LANGFUSE_HOST — e.g. an
            # unset CI secret injected as "" — must fall back to the default, not
            # override it with an empty string (which yields a scheme-less URL and
            # a flood of "No scheme supplied" export errors).
            host=os.environ.get("LANGFUSE_HOST") or _DEFAULT_LANGFUSE_HOST,
        )
        _observe_impl = _lf_observe
        _ENABLED = True
        logger.info("Langfuse tracing ENABLED.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Langfuse init failed (%s); tracing disabled.", exc)
        _ENABLED = False


def tracing_enabled() -> bool:
    """True iff Langfuse tracing is live."""
    return _ENABLED


# ---------------------------------------------------------------------------
# @observe — trace a pipeline stage. Real span when enabled, no-op otherwise.
# ---------------------------------------------------------------------------

def observe(
    name: str | None = None,
    *,
    as_type: str = "span",  # "span" | "generation"
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """Wrap a stage so it becomes a traced observation.

    Use ``as_type="generation"`` on LLM calls so Langfuse treats token/cost
    fields specially. When disabled, returns the original function untouched.
    """
    def decorator(fn: F) -> F:
        if not _ENABLED or _observe_impl is None:
            return fn
        wrapped = _observe_impl(
            name=name or fn.__name__,
            as_type=as_type,
            capture_input=capture_input,
            capture_output=capture_output,
        )(fn)

        @functools.wraps(fn)
        def outer(*args: Any, **kwargs: Any) -> Any:
            return wrapped(*args, **kwargs)

        return outer  # type: ignore[return-value]

    return decorator


def record(**attributes: Any) -> None:
    """Attach metadata to the current span. No-op when disabled."""
    if not _ENABLED or _langfuse is None:
        return
    try:
        _langfuse.update_current_span(metadata=dict(attributes))
    except Exception as exc:  # pragma: no cover
        logger.debug("record() failed: %s", exc)


def record_usage(
    *,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Attach model + token usage to the current generation. Langfuse computes
    cost from model + tokens, so no price table lives in the repo.
    """
    if not _ENABLED or _langfuse is None:
        return
    usage: dict[str, Any] = {}
    if input_tokens is not None:
        usage["input"] = input_tokens
    if output_tokens is not None:
        usage["output"] = output_tokens
    try:
        _langfuse.update_current_generation(model=model, usage_details=usage or None)
    except Exception as exc:  # pragma: no cover
        logger.debug("record_usage() failed: %s", exc)


def score(name: str, value: float, *, comment: str | None = None) -> None:
    """Attach a numeric eval score to the current trace. No-op when disabled."""
    if not _ENABLED or _langfuse is None:
        return
    try:
        _langfuse.score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:  # pragma: no cover
        logger.debug("score() failed: %s", exc)


@contextmanager
def trace_context(
    name: str,
    *,
    session_id: str | None = None,
    tags: list[str] | None = None,
    **metadata: Any,
) -> Iterator[None]:
    """Open a timed root span for the whole request and name/tag the trace.

    Using ``start_as_current_observation(as_type="span")`` (v3 SDK) matters:
    it creates a real observation with start/end timestamps, so Langfuse can
    compute *trace-level* latency — not just per-generation latency. All the
    @observe-decorated stages nest under this root automatically via OTel
    context propagation. Without a root span the trace envelope has no duration
    and the trace latency dashboard stays empty even though generations are
    timed. No-op when disabled; never raises into the pipeline.
    """
    if not _ENABLED or _langfuse is None:
        yield
        return

    # Try to open a real timed root span. If the SDK surface differs, fall back
    # to a null context so we still annotate the trace without losing tracing.
    # Deciding the context manager up front (rather than wrapping the yield in a
    # try/except) avoids any double-yield on body exceptions.
    try:
        root_cm: Any = _langfuse.start_as_current_observation(as_type="span", name=name)
    except Exception as exc:  # pragma: no cover
        logger.debug("start_as_current_observation unavailable (%s); annotating only.", exc)
        root_cm = nullcontext()

    with root_cm:
        try:
            _langfuse.update_current_trace(
                name=name,
                session_id=session_id,
                tags=tags,
                metadata=dict(metadata) or None,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("update_current_trace failed: %s", exc)
        yield


def flush() -> None:
    """Flush buffered Langfuse events. Call before a short-lived process exits."""
    if not _ENABLED or _langfuse is None:
        return
    try:
        _langfuse.flush()
    except Exception as exc:  # pragma: no cover
        logger.debug("flush() failed: %s", exc)


# ---------------------------------------------------------------------------
# Phase 2 — local metrics sink. Always on (unless RAG_METRICS_PATH is empty),
# independent of Langfuse. One JSON line per request → percentile & rate math.
# ---------------------------------------------------------------------------

def _metrics_path() -> Path | None:
    raw = os.environ.get("RAG_METRICS_PATH", "metrics/requests.jsonl")
    if not raw.strip():
        return None
    return Path(raw)


def metrics_sink_path() -> Path | None:
    """The path the request sink writes to, or None if disabled.

    Set ``RAG_METRICS_PATH`` empty to disable the sink entirely (e.g. if you ever
    want a fully silent deployment). Public so the Streamlit admin panel can read
    the same file the pipeline writes.
    """
    return _metrics_path()


def metrics_enabled() -> bool:
    """True iff the local metrics sink is active."""
    return _metrics_path() is not None


def emit_request_metrics(record_dict: dict[str, Any]) -> None:
    """Append one request's metrics as a JSON line. Fail-safe.

    Fields written by RequestTimer.finish(); see below. Timestamped in UTC so
    metrics can be windowed over time.
    """
    path = _metrics_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record_dict = {"ts": datetime.now(UTC).isoformat(), **record_dict}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover
        logger.debug("emit_request_metrics() failed: %s", exc)


class RequestTimer:
    """Collects the per-request SRE signals for Phase 2, then emits them to the
    local sink and (if enabled) as Langfuse scores/metadata.

    Usage (in a query entry point)::

        with RequestTimer(mode="single_doc", question=q) as t:
            result = pipeline.ask(q)
            t.set_result(
                declined=result.declined,
                n_citations=len(result.citations),
                n_context_chunks=len(contexts),
            )

    Signals captured:
      * latency_ms          — wall-clock for the whole request
      * llm_calls           — count of model calls (rewrite/generate/judge)
      * declined            — did the system decline?
      * error               — did it raise? (failure-rate numerator)
      * cited / grounded    — citation-coverage numerator
    """

    def __init__(self, *, mode: str, question: str = "") -> None:
        self.mode = mode
        self.question = question[:120]
        self._t0 = 0.0
        self.latency_ms: float = 0.0
        self.llm_calls: int = 0
        self.declined: bool = False
        self.n_citations: int = 0
        self.n_context_chunks: int = 0
        self.error: str | None = None

    def __enter__(self) -> RequestTimer:
        self._t0 = time.perf_counter()
        _active_timer.set(self)
        return self

    def bump_llm_calls(self, n: int = 1) -> None:
        self.llm_calls += n

    def set_result(
        self,
        *,
        declined: bool,
        n_citations: int,
        n_context_chunks: int,
    ) -> None:
        self.declined = declined
        self.n_citations = n_citations
        self.n_context_chunks = n_context_chunks

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        self.latency_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc is not None:
            self.error = f"{exc_type.__name__}: {exc}"[:200]
        _active_timer.set(None)

        # An answered request is "grounded" when it produced >=1 inline citation.
        # Declines and errors are excluded from the coverage denominator.
        grounded: bool | None
        if self.error is not None or self.declined:  # noqa: SIM108
            grounded = None
        else:
            grounded = self.n_citations > 0

        payload = {
            "mode": self.mode,
            "question": self.question,
            "latency_ms": round(self.latency_ms, 1),
            "llm_calls": self.llm_calls,
            "declined": self.declined,
            "error": self.error,
            "n_citations": self.n_citations,
            "n_context_chunks": self.n_context_chunks,
            "grounded": grounded,
            "config_version": os.environ.get("RAG_CONFIG_VERSION", ""),
        }
        emit_request_metrics(payload)

        # Mirror the key signals into Langfuse when enabled.
        record(
            latency_ms=payload["latency_ms"],
            llm_calls=self.llm_calls,
            declined=self.declined,
            error=self.error,
            grounded=grounded,
        )
        if grounded is not None:
            score("citation_coverage", 1.0 if grounded else 0.0)
        score("request_error", 1.0 if self.error else 0.0)

        # Never suppress a real exception — return False so it propagates.
        return False


# A tiny context var so LLM wrappers can find the active timer to bump call
# counts without threading the timer through every function signature.
_active_timer: ContextVar[RequestTimer | None] = ContextVar(
    "_active_timer", default=None
)


def count_llm_call(n: int = 1) -> None:
    """Increment the active request's LLM-call counter, if one is open."""
    t = _active_timer.get()
    if t is not None:
        t.bump_llm_calls(n)


__all__ = [
    "observe",
    "record",
    "record_usage",
    "score",
    "trace_context",
    "flush",
    "tracing_enabled",
    "metrics_enabled",
    "metrics_sink_path",
    "RequestTimer",
    "count_llm_call",
    "emit_request_metrics",
]
