"""Tests for the observability layer.

These run keyless and offline (like the rest of the suite): with no LANGFUSE_*
env vars, tracing is a no-op, and the local metrics sink is redirected to a
tmp file. We assert the two things that actually matter:

  1. The no-op path is truly inert — decorators and helpers don't alter behavior.
  2. The Phase 2 math is correct — percentiles, citation coverage (declines and
     errors excluded), failure rate, and the gate's pass/fail decision.
"""

from __future__ import annotations

import json

import pytest

from rag_revops import metrics_report as mr
from rag_revops import observability as obs


@pytest.fixture(autouse=True)
def _isolate_sink(tmp_path, monkeypatch):
    """Point the metrics sink at a tmp file and force tracing OFF for tests.

    tracing_enabled() reflects a module-level flag computed once at import time
    from the environment. A developer running the suite locally may have real
    LANGFUSE_* keys set (so the module imported with tracing ON), and deleting
    the env vars here can't retroactively flip an already-imported flag. So we
    patch the flag directly — this guarantees the no-op path is what's under
    test, whether or not the developer has keys configured.
    """
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(obs, "_ENABLED", False)
    monkeypatch.setattr(obs, "_langfuse", None)
    sink = tmp_path / "requests.jsonl"
    monkeypatch.setenv("RAG_METRICS_PATH", str(sink))
    return sink


def test_tracing_disabled_without_keys():
    # With the flag patched off (see fixture), tracing must report disabled and
    # the helpers must all be inert — that's the contract this suite verifies.
    assert obs.tracing_enabled() is False


def test_observe_is_transparent_noop():
    """A wrapped function returns identically and runs its body exactly once."""
    calls = []

    @obs.observe("stage")
    def stage(x, y):
        obs.record(k="v")            # no-op, must not raise
        obs.record_usage(model="m", input_tokens=1, output_tokens=1)
        obs.score("s", 0.5)
        calls.append((x, y))
        return x + y

    assert stage(2, 3) == 5
    assert calls == [(2, 3)]
    # When disabled, the decorator returns the original function object.
    assert stage.__name__ == "stage"


def test_request_timer_writes_a_record(_isolate_sink):
    with obs.RequestTimer(mode="single_doc", question="hello") as t:
        obs.count_llm_call()
        obs.count_llm_call()
        t.set_result(declined=False, n_citations=2, n_context_chunks=5)

    lines = _isolate_sink.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["mode"] == "single_doc"
    assert rec["llm_calls"] == 2
    assert rec["grounded"] is True
    assert rec["error"] is None
    assert "ts" in rec


def test_request_timer_does_not_swallow_exceptions(_isolate_sink):
    with (
        pytest.raises(ValueError),
        obs.RequestTimer(mode="single_doc", question="boom"),
    ):
        obs.count_llm_call()
        raise ValueError("boom")

    rec = json.loads(_isolate_sink.read_text(encoding="utf-8").strip())
    assert rec["error"] is not None
    assert rec["grounded"] is None            # errors excluded from coverage


def test_grounded_is_none_for_declines(_isolate_sink):
    with obs.RequestTimer(mode="single_doc", question="q") as t:
        t.set_result(declined=True, n_citations=0, n_context_chunks=0)
    rec = json.loads(_isolate_sink.read_text(encoding="utf-8").strip())
    assert rec["grounded"] is None            # declines excluded from coverage


# --- Phase 2 math -----------------------------------------------------------

def test_percentile_nearest_rank():
    vals = [10, 20, 30, 40, 50]
    assert mr._percentile(vals, 50) == 30
    assert mr._percentile(vals, 95) == 50
    assert mr._percentile([], 50) is None
    assert mr._percentile([7], 95) == 7


def test_compute_rates_and_coverage():
    records = [
        # answered + cited  → grounded True
        {"latency_ms": 100, "llm_calls": 2, "declined": False,
         "error": None, "grounded": True},
        # answered, no citation → grounded False (counts against coverage)
        {"latency_ms": 200, "llm_calls": 1, "declined": False,
         "error": None, "grounded": False},
        # decline → excluded from coverage denominator
        {"latency_ms": 50, "llm_calls": 0, "declined": True,
         "error": None, "grounded": None},
        # error → failure, excluded from coverage
        {"latency_ms": 300, "llm_calls": 1, "declined": False,
         "error": "ValueError: x", "grounded": None},
    ]
    m = mr.compute(records)
    assert m["n_requests"] == 4
    assert m["failure_rate"] == pytest.approx(0.25)
    assert m["decline_rate"] == pytest.approx(0.25)
    # coverage: 1 grounded of 2 answered-and-scored = 0.5
    assert m["citation_coverage"] == pytest.approx(0.5)
    assert m["n_answered_scored_for_coverage"] == 2
    assert m["llm_calls"]["total"] == 4


def test_gate_pass_and_fail():
    records = [
        {"latency_ms": 100, "llm_calls": 1, "declined": False,
         "error": None, "grounded": True}
        for _ in range(9)
    ] + [
        {"latency_ms": 100, "llm_calls": 1, "declined": False,
         "error": None, "grounded": False}
    ]
    m = mr.compute(records)          # coverage = 0.9, failure = 0.0
    # Lenient: passes.
    assert mr.apply_gate(
        m, min_coverage=0.85, max_failure_rate=0.05, max_p95_ms=1_000
    ) == 0
    # Strict coverage: fails.
    assert mr.apply_gate(
        m, min_coverage=0.95, max_failure_rate=0.05, max_p95_ms=1_000
    ) == 1


def test_gate_skips_unmeasurable_thresholds():
    """No traffic → thresholds warn and skip rather than failing on absence."""
    m = mr.compute([])
    assert mr.apply_gate(
        m, min_coverage=0.9, max_failure_rate=0.05, max_p95_ms=1_000
    ) == 0


# --- admin panel gating -----------------------------------------------------

class _FakeSecrets(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeStreamlit:
    """Minimal Streamlit stand-in that records what the panel tried to render."""

    def __init__(self, secrets, pw_input):
        self._secrets = _FakeSecrets(secrets)
        self._pw = pw_input
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.metrics_rendered = False

    @property
    def secrets(self):
        return self._secrets

    @property
    def sidebar(self):
        return _FakeCtx()

    def divider(self):
        pass

    def expander(self, *a, **k):
        return _FakeCtx()

    def text_input(self, *a, **k):
        return self._pw

    def error(self, m):
        self.errors.append(m)

    def success(self, m):
        self.successes.append(m)

    def info(self, m):
        pass

    def caption(self, m):
        pass

    def selectbox(self, *a, **k):
        return "All time"

    def columns(self, n):
        self.metrics_rendered = True
        return [self._Col() for _ in range(n)]

    def json(self, obj):
        pass

    class _Col:
        def metric(self, *a, **k):
            pass


def _panel():
    from rag_revops import admin_panel
    return admin_panel


def test_admin_panel_silent_without_password():
    st = _FakeStreamlit(secrets={}, pw_input="")
    _panel().render_admin_panel(st)
    assert not st.metrics_rendered and not st.errors


def test_admin_panel_rejects_wrong_password():
    st = _FakeStreamlit(secrets={"admin_password": "secret"}, pw_input="nope")
    _panel().render_admin_panel(st)
    assert st.errors and not st.metrics_rendered


def test_admin_panel_unlocks_with_correct_password(_isolate_sink):
    # Seed one request so there's something to render.
    with obs.RequestTimer(mode="single_doc", question="q") as t:
        obs.count_llm_call()
        t.set_result(declined=False, n_citations=1, n_context_chunks=3)

    st = _FakeStreamlit(secrets={"admin_password": "secret"}, pw_input="secret")
    _panel().render_admin_panel(st)
    assert st.successes and st.metrics_rendered
