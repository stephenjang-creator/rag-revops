"""Tests for the provider-flexible query rewriter.

Rewriting defaults to Cohere (reusing the app's Cohere key) and falls back to
Anthropic when no Cohere key is present. Both clients are stubbed and secrets are
controlled via env, so these run with no network. Metrics sink disabled.
"""

from __future__ import annotations

import pytest

from rag_revops.config import load_secrets, load_settings
from rag_revops.query_rewrite import QueryRewriter


# --- Cohere stub -------------------------------------------------------------
class _CohereResp:
    def __init__(self, text: str):
        self.text = text
        self.meta = None


class _CohereClient:
    def __init__(self, text: str):
        self._text = text

    def chat(self, **kwargs):
        return _CohereResp(self._text)


# --- Anthropic stub ----------------------------------------------------------
class _Block:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Usage:
    input_tokens = 5
    output_tokens = 5


class _AnthResp:
    def __init__(self, text: str):
        self.content = [_Block(text)]
        self.usage = _Usage()


class _AnthMessages:
    def __init__(self, text: str):
        self._text = text

    def create(self, **kwargs):
        return _AnthResp(self._text)


class _AnthClient:
    def __init__(self, text: str):
        self.messages = _AnthMessages(text)


@pytest.fixture(autouse=True)
def _no_metrics_sink(monkeypatch):
    monkeypatch.setenv("RAG_METRICS_PATH", "")


@pytest.fixture(autouse=True)
def _clear_secrets_cache():
    load_secrets.cache_clear()
    yield
    load_secrets.cache_clear()


def _rewriter() -> QueryRewriter:
    return QueryRewriter(load_settings())


def test_uses_cohere_when_cohere_key_present(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "co-x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load_secrets.cache_clear()

    r = _rewriter()
    r._cohere = _CohereClient(
        "Can either party terminate for convenience, and with how much notice?"
    )
    out = r.rewrite("can either party do a TFC")

    assert out.changed is True
    assert out.rewritten == (
        "Can either party terminate for convenience, and with how much notice?"
    )
    assert r._anthropic is None  # never touched Anthropic


def test_falls_back_to_anthropic_when_no_cohere_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    load_secrets.cache_clear()

    r = _rewriter()
    r._anthropic = _AnthClient("Rewritten via Anthropic")
    out = r.rewrite("whats the MFN")

    assert out.changed is True
    assert out.rewritten == "Rewritten via Anthropic"
    assert r._cohere is None  # never touched Cohere


def test_no_keys_returns_original_unchanged(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load_secrets.cache_clear()

    out = _rewriter().rewrite("IP ownership?")
    assert out.changed is False
    assert out.rewritten == "IP ownership?"


def test_empty_question_is_noop(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "co-x")
    load_secrets.cache_clear()

    out = _rewriter().rewrite("   ")
    assert out.changed is False


def test_empty_rewrite_falls_back_to_original(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "co-x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load_secrets.cache_clear()

    r = _rewriter()
    r._cohere = _CohereClient("")  # degenerate/empty rewrite
    out = r.rewrite("can either party do a TFC")

    assert out.changed is False
    assert out.rewritten == "can either party do a TFC"
