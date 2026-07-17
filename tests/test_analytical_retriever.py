"""Tests for AnalyticalRetriever coverage — it must judge the ENTIRE contract set
by default, with optional caps for cost control.

Live embedding/rerank are stubbed, so these lock in the pure orchestration:
default (pool_size=0, max_contracts=0) fetches the whole collection and returns
every contract; positive caps trim coverage.
"""

from __future__ import annotations

from rag_revops.analytical import AnalyticalRetriever
from rag_revops.config import load_settings
from rag_revops.rerank import RerankedChunk
from rag_revops.vectorstore import RetrievedChunk


def _corpus(n_contracts: int, chunks_each: int = 2) -> list[RetrievedChunk]:
    # Contract-major order so each contract's first chunk is its representative.
    chunks: list[RetrievedChunk] = []
    for c in range(n_contracts):
        doc = f"C{c:02d}"
        for j in range(chunks_each):
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{doc}-{j}", doc_id=doc, text=f"{doc} chunk {j}",
                    score=1.0 - (len(chunks) * 0.001),
                    metadata={"doc_id": doc, "source_path": f"{doc}.txt"},
                )
            )
    return chunks


class FakeStore:
    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks

    def count(self) -> int:
        return len(self._chunks)

    def query(self, query_embedding, top_k, doc_id=None):
        return self._chunks[:top_k]


class FakeEmbedder:
    def embed_query(self, text: str):
        return [0.0, 0.0, 0.0]


class FakeReranker:
    """Preserves input order, assigning descending scores (no real cross-encoder)."""

    def rerank(self, query, chunks, top_n=None):
        return [
            RerankedChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text,
                score=1.0 - i * 0.01, metadata=c.metadata,
            )
            for i, c in enumerate(chunks)
        ]


def _retriever(chunks, **analytical_overrides) -> AnalyticalRetriever:
    settings = load_settings().model_copy(deep=True)
    for k, v in analytical_overrides.items():
        setattr(settings.analytical, k, v)
    return AnalyticalRetriever(settings, FakeStore(chunks), FakeEmbedder(), FakeReranker())


def test_default_judges_every_contract():
    chunks = _corpus(12)
    matches = _retriever(chunks).retrieve("which contracts cap liability?")
    assert len(matches) == 12
    assert {m.doc_id for m in matches} == {f"C{c:02d}" for c in range(12)}


def test_default_fetches_entire_collection_even_beyond_old_2000_cap():
    # 1100 contracts x 2 chunks = 2200 chunks — above the retired 2000 pool cap.
    # With pool_size=0 the whole collection is fetched, so every contract surfaces.
    chunks = _corpus(1100)
    matches = _retriever(chunks).retrieve("q")
    assert len(matches) == 1100


def test_positive_max_contracts_caps_candidates():
    chunks = _corpus(12)
    matches = _retriever(chunks, max_contracts=5).retrieve("q")
    assert len(matches) == 5


def test_positive_pool_size_can_limit_coverage():
    # pool_size=4 fetches only the first 4 chunks (contracts C00, C01 — 2 each),
    # so only those contracts are represented.
    chunks = _corpus(12)
    matches = _retriever(chunks, pool_size=4).retrieve("q")
    assert {m.doc_id for m in matches} == {"C00", "C01"}
