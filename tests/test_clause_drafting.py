"""Tests for the clause drafter — suggests new clause language from retrieved
examples, citing the sources it used.

The Anthropic call is stubbed (no network / no key), so these lock in the pure
logic: which examples end up cited ([n] parsing), the no-options decline path,
and the explicit-decline-message path. The metrics sink is disabled per-test.
"""

from __future__ import annotations

import pytest

from rag_revops.clause_drafting import (
    DECLINE_MESSAGE,
    ClauseDrafter,
    decline_reason,
    strip_citations,
)
from rag_revops.clause_finder import ClauseOption
from rag_revops.config import load_settings


class _Block:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Usage:
    input_tokens = 10
    output_tokens = 20


class _Resp:
    def __init__(self, text: str):
        self.content = [_Block(text)]
        self.usage = _Usage()


class _Messages:
    def __init__(self, text: str):
        self._text = text

    def create(self, **kwargs):
        return _Resp(self._text)


class _Client:
    def __init__(self, text: str):
        self.messages = _Messages(text)


def _opt(doc_id: str, score: float = 0.9) -> ClauseOption:
    return ClauseOption(
        doc_id=doc_id, source_path=f"{doc_id}.txt", text=f"{doc_id} clause text",
        score=score, chunk_id=f"{doc_id}-c0",
    )


@pytest.fixture(autouse=True)
def _no_metrics_sink(monkeypatch):
    monkeypatch.setenv("RAG_METRICS_PATH", "")


def _drafter(canned_text: str) -> ClauseDrafter:
    d = ClauseDrafter(load_settings())
    d._client = _Client(canned_text)  # inject stub; _get_client returns it as-is
    return d


def test_draft_keeps_only_cited_examples():
    options = [_opt("A"), _opt("B"), _opt("C")]
    # Draft references [1] and [3] but not [2].
    draft = _drafter("Either party may terminate for convenience [1] on notice [3].")
    result = draft.draft("termination for convenience", options)

    assert result.declined is False
    assert [o.doc_id for o in result.citations] == ["A", "C"]
    assert result.options == options  # all retrieved examples retained for reference


def test_no_options_declines_without_model_call():
    # No client set at all — if draft() called the model, this would raise.
    d = ClauseDrafter(load_settings())
    result = d.draft("indemnification", [])

    assert result.declined is True
    assert result.draft == DECLINE_MESSAGE
    assert result.citations == []


def test_explicit_decline_message_is_flagged():
    result = _drafter(DECLINE_MESSAGE).draft("something obscure", [_opt("A")])

    assert result.declined is True
    assert result.citations == []


def test_decline_detected_when_message_is_a_prefix_with_explanation():
    # Real model behavior: decline sentence + a follow-on explanation. Must still
    # be treated as a decline (not rendered as a clause / copy block).
    text = (
        f"{DECLINE_MESSAGE}\n\nThe two examples retrieved ([1] and [2]) both "
        "originate from the same franchise agreement and don't generalize."
    )
    result = _drafter(text).draft("non-compete", [_opt("A"), _opt("B")])

    assert result.declined is True
    assert result.citations == []


def test_decline_reason_extracts_explanation_after_the_sentence():
    text = f"{DECLINE_MESSAGE}\n\nBoth examples are franchise-specific."
    assert decline_reason(text) == "Both examples are franchise-specific."


def test_decline_reason_empty_when_only_the_sentence():
    assert decline_reason(DECLINE_MESSAGE) == ""


def test_strip_citations_removes_markers_and_tidies_spacing():
    text = "Either party may terminate for convenience [1] upon 30 days' notice [2]."
    assert strip_citations(text) == (
        "Either party may terminate for convenience upon 30 days' notice."
    )


def test_strip_citations_handles_grouped_and_leading_markers():
    text = "[1] The Receiving Party shall keep [2, 3] Confidential Information secret."
    assert strip_citations(text) == (
        "The Receiving Party shall keep Confidential Information secret."
    )


def test_strip_citations_noop_without_markers():
    text = "Either party may terminate this Agreement for convenience."
    assert strip_citations(text) == text


def test_strip_citations_clears_parenthetical_left_holding_only_markers():
    text = "The two examples ([1] and [2]) both cap liability."
    assert strip_citations(text) == "The two examples both cap liability."
