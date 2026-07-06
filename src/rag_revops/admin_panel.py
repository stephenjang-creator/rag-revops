"""Password-gated admin panel for the Streamlit app.

Renders a live operational-metrics view in the sidebar — but only for someone
who enters the admin password configured in Streamlit secrets. Reviewers who
don't have the password see nothing at all (no panel, no hint it exists).

Wiring (in app.py sidebar)::

    from rag_revops.admin_panel import render_admin_panel
    render_admin_panel()

Configure the password in Streamlit Cloud → Settings → Secrets (or a local
.streamlit/secrets.toml)::

    admin_password = "something-only-you-know"

Design notes
------------
* **Silent by default.** If no ``admin_password`` is configured, the panel does
  not render and does not announce itself. The metrics sink still collects data
  (that's the point — you want production traffic), it's just not *viewable* in
  the deployed app without the password.
* **Metrics collection is independent of this panel.** The panel only *reads* the
  sink that ``RequestTimer`` already writes. Turning the panel off doesn't turn
  collection off, and vice-versa.
* **No Langfuse required.** This reads the local JSONL sink, so it works on the
  deployed app even though Langfuse tracing is off there.
"""

from __future__ import annotations

from typing import Any

from . import metrics_report as mr
from .observability import metrics_sink_path, tracing_enabled


def _get_admin_password(st) -> str | None:
    """Read the admin password from Streamlit secrets, tolerating its absence.

    Accessing a missing key in st.secrets raises, so guard it — a demo without
    the secret configured must not crash, it must simply not show the panel.
    """
    try:
        pw = st.secrets.get("admin_password", "")
    except Exception:
        pw = ""
    return pw or None


def render_admin_panel(st=None) -> None:
    """Render the password-gated admin metrics panel in the sidebar.

    ``st`` is injected for testability; in the app just call
    ``render_admin_panel()`` and it imports streamlit itself.
    """
    if st is None:  # pragma: no cover - trivial import shim
        import streamlit as st  # type: ignore

    admin_password = _get_admin_password(st)
    if admin_password is None:
        # No password configured → panel is invisible. Metrics still collect.
        return

    with st.sidebar:
        st.divider()
        with st.expander("🔒 Admin", expanded=False):
            entered = st.text_input(
                "Admin password",
                type="password",
                key="_admin_pw",
                help="Operational metrics view. Reviewers can ignore this.",
            )
            if not entered:
                return
            if entered != admin_password:
                st.error("Incorrect password.")
                return

            st.success("Admin unlocked.")
            _render_metrics(st)


def _render_metrics(st) -> None:
    """Read the request sink and render the live operational metrics."""
    sink = metrics_sink_path()
    if sink is None:
        st.info("Metrics sink is disabled (RAG_METRICS_PATH is empty).")
        return

    # Optional time window.
    window = st.selectbox(
        "Window",
        options=["All time", "Last 60 min", "Last 24 h"],
        index=0,
        key="_admin_window",
    )
    since_min = {"All time": None, "Last 60 min": 60, "Last 24 h": 24 * 60}[window]

    records = mr.load_records(sink, since_min)
    if not records:
        st.info(
            "No requests recorded yet in this window. "
            "Run a query, then reopen this panel."
        )
        _render_status_line(st)
        return

    m = mr.compute(records)
    _render_metric_cards(st, m)
    _render_status_line(st)

    with st.expander("Raw metrics (JSON)", expanded=False):
        st.json(m)


def _render_metric_cards(st, m: dict[str, Any]) -> None:
    lat = m["latency_ms"]

    st.caption(f"Requests observed: **{m['n_requests']}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Latency p50", _ms(lat["p50"]))
    c2.metric("Latency p95", _ms(lat["p95"]))
    c3.metric("Latency p99", _ms(lat["p99"]))

    c4, c5, c6 = st.columns(3)
    c4.metric("Citation coverage", _pct(m["citation_coverage"]))
    c5.metric("Failure rate", _pct(m["failure_rate"]))
    c6.metric("Decline rate", _pct(m["decline_rate"]))

    c7, c8 = st.columns(2)
    c7.metric("LLM calls / req (p50)", _num(m["llm_calls"]["p50"]))
    c8.metric("LLM calls / req (max)", _num(m["llm_calls"]["max"]))

    # The mean is shown next to p95 deliberately: it hides the tail.
    if lat["mean"] is not None and lat["p95"] is not None:
        st.caption(
            f"Mean latency {lat['mean']:.0f} ms vs p95 {lat['p95']:.0f} ms — "
            "the gap is your tail."
        )


def _render_status_line(st) -> None:
    st.caption(
        f"Langfuse tracing: **{'on' if tracing_enabled() else 'off'}**  ·  "
        "metrics sink: **on**"
    )


# --- tiny formatters --------------------------------------------------------

def _ms(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f} ms"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _num(v: float | int | None) -> str:
    return "—" if v is None else (f"{v:.0f}" if isinstance(v, float) else str(v))
