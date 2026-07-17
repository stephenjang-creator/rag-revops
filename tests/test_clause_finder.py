"""Tests for the clause-language finder (reranker-driven precision retrieval).

The finder needs live embedding + rerank APIs, so we lock in its pure
orchestration logic — best-passage-per-contract dedup, the relevance floor, and
the max-options cap — with injected fakes and no network. The metrics sink is
disabled per-test so nothing is written to the repo.
"""

from __future__ import annotations

import pytest

from rag_revops.clause_finder import ClauseFinder
from rag_revops.config import load_settings
from rag_revops.rerank import RerankedChunk
from rag_revops.vectorstore import RetrievedChunk


class FakeStore:
    def __init__(self, n: int):
        self._n = n

    def count(self) -> int:
        return self._n

    def query(self, query_embedding, top_k, doc_id=None):
        # Content is irrelevant — the fake reranker returns a preset ordering.
        return [
            RetrievedChunk(chunk_id=f"p{i}", doc_id="pool", text="t", score=0.5,
                           metadata={})
            for i in range(min(top_k, self._n))
        ]


class FakeEmbedder:
    def embed_query(self, text: str):
        return [0.0, 0.0, 0.0]


class FakeReranker:
    """Returns preset reranked chunks (already best-first), ignoring the query."""

    def __init__(self, ranked: list[RerankedChunk]):
        self._ranked = ranked

    def rerank(self, query, chunks, top_n=None):
        return self._ranked


def _rc(chunk_id: str, doc_id: str, score: float) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=chunk_id, doc_id=doc_id, text=f"{doc_id} clause language",
        score=score, metadata={"source_path": f"{doc_id}.txt"},
    )


@pytest.fixture(autouse=True)
def _no_metrics_sink(monkeypatch):
    # Empty RAG_METRICS_PATH disables the local sink, so find() writes nothing.
    monkeypatch.setenv("RAG_METRICS_PATH", "")


def _finder(ranked: list[RerankedChunk], pool_n: int = 10) -> ClauseFinder:
    settings = load_settings()
    return ClauseFinder(settings, FakeStore(pool_n), FakeEmbedder(), FakeReranker(ranked))


def test_dedupes_to_best_passage_per_contract():
    ranked = [
        _rc("a1", "A", 0.90),
        _rc("a2", "A", 0.80),  # same contract, lower — dropped
        _rc("b1", "B", 0.70),
    ]
    result = _finder(ranked).find("termination for convenience")
    assert [o.doc_id for o in result.options] == ["A", "B"]
    assert result.options[0].chunk_id == "a1"  # A's best passage is kept


def test_relevance_floor_excludes_low_scores():
    # Default floor is 0.30 (config.ClauseConfig.min_relevance).
    ranked = [_rc("a1", "A", 0.90), _rc("b1", "B", 0.10)]
    result = _finder(ranked).find("q")
    assert [o.doc_id for o in result.options] == ["A"]


def test_max_options_cap_defaults_to_six():
    # Eight distinct contracts, all above the floor — capped at max_options (6).
    ranked = [_rc(f"c{i}", f"D{i}", 0.9) for i in range(8)]
    result = _finder(ranked, pool_n=8).find("q")
    assert len(result.options) == 6
    assert result.n_considered == 8


def test_empty_store_returns_no_options():
    result = _finder([_rc("a1", "A", 0.9)], pool_n=0).find("q")
    assert result.options == []
    assert result.n_considered == 0
