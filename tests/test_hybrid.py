"""Tests for Phase 2 hybrid retrieval: RRF fusion and the BM25 tokenizer."""

from __future__ import annotations

from rag_revops.bm25 import tokenize
from rag_revops.hybrid import reciprocal_rank_fusion
from rag_revops.vectorstore import RetrievedChunk


def _mk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, doc_id="d", text=f"text-{cid}", score=0.0, metadata={})


def test_chunk_in_both_lists_ranks_first():
    dense = [_mk("A"), _mk("B"), _mk("D")]
    lexical = [_mk("B"), _mk("C"), _mk("E")]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=5)
    assert fused[0].chunk_id == "B"


def test_rrf_score_matches_formula():
    dense = [_mk("A"), _mk("B")]
    lexical = [_mk("B")]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=5)
    b = next(c for c in fused if c.chunk_id == "B")
    expected = 1 / (60 + 2) + 1 / (60 + 1)
    assert abs(b.score - expected) < 1e-9
    assert b.dense_rank == 2 and b.lexical_rank == 1


def test_single_list_rank_beats_lower_single_list_rank():
    dense = [_mk("A")]
    lexical = [_mk("X"), _mk("C")]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=5)
    ids = [c.chunk_id for c in fused]
    assert ids.index("A") < ids.index("C")


def test_top_k_truncates():
    dense = [_mk(str(i)) for i in range(10)]
    lexical = [_mk(str(i)) for i in range(10, 20)]
    fused = reciprocal_rank_fusion(dense, lexical, k=60, top_k=3)
    assert len(fused) == 3


def test_tokenizer_preserves_contract_terms():
    toks = tokenize("Payment terms are Net 30 per Section 7.2 of the Agreement.")
    assert "30" in toks
    assert "7.2" in toks
    assert "net" in toks
    assert "section" in toks


def test_tokenizer_lowercases():
    assert tokenize("Confidential INFORMATION") == ["confidential", "information"]