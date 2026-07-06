"""Tests for the Cohere reranker's result mapping.

The live rerank call needs the network + a key, so we stub the Cohere client and
test the part that's easy to get wrong: mapping the reranker's reordered response
indices back to the correct source chunks, sorting, and metadata preservation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


def _make_reranker(rerank_response):
    """Build a CohereReranker whose client returns a canned rerank response."""
    from types import SimpleNamespace

    from rag_revops.config import RerankConfig
    from rag_revops.rerank import CohereReranker

    cfg = RerankConfig(max_retries=1, backoff_base_s=0.0, backoff_cap_s=0.0)
    r = CohereReranker(cfg)
    # Inject a stub client so no key / network is needed (client is lazy).
    r._client = SimpleNamespace(rerank=lambda **kwargs: rerank_response)
    return r


class _Item:
    def __init__(self, index: int, score: float):
        self.index = index
        self.relevance_score = score


class _Resp:
    def __init__(self, items):
        self.results = items


@pytest.fixture
def chunks():
    return [_Chunk(f"c{i}", "d", f"passage {i}", {"i": i}) for i in range(5)]


def test_reorders_and_maps_indices_to_source(chunks):
    resp = _Resp([_Item(3, 0.91), _Item(0, 0.55), _Item(4, 0.12)])
    out = _make_reranker(resp).rerank("q", chunks)
    assert [c.chunk_id for c in out] == ["c3", "c0", "c4"]
    assert abs(out[0].score - 0.91) < 1e-9


def test_sorted_descending(chunks):
    resp = _Resp([_Item(0, 0.2), _Item(1, 0.9), _Item(2, 0.5)])
    out = _make_reranker(resp).rerank("q", chunks)
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_metadata_follows_source_chunk(chunks):
    resp = _Resp([_Item(3, 0.8)])
    out = _make_reranker(resp).rerank("q", chunks)
    assert out[0].metadata == {"i": 3}


def test_empty_input_returns_empty(chunks):
    # Should short-circuit without calling the client.
    r = _make_reranker(_Resp([]))
    assert r.rerank("q", []) == []
