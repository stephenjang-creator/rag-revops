"""Tests for the clause drafter — suggests new clause language from retrieved
examples, citing the sources it used.

The Anthropic call is stubbed (no network / no key), so these lock in the pure
logic: which examples end up cited ([n] parsing), the no-options decline path,
and the explicit-decline-message path. The metrics sink is disabled per-test.
"""

from __future__ import annotations

import pytest

from rag_revops.clause_drafting import DECLINE_MESSAGE, ClauseDrafter
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
