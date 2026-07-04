"""Hybrid retrieval: fuse dense (vector) and lexical (BM25) results.

The two retrievers score on incompatible scales — cosine similarity is ~[0,1],
BM25 is an unbounded positive score — so we can't just add them. Reciprocal Rank
Fusion (RRF) sidesteps this by fusing on *rank position* instead of raw score:

    rrf_score(chunk) = sum over lists of 1 / (k + rank_in_list)

A chunk that ranks highly in either list floats up; a chunk that ranks decently
in *both* floats highest. `k` (the RRF constant, conventionally 60) damps the
influence of very high ranks so the tail still contributes.

Each retriever fetches `fetch_k` candidates; we fuse and return the top `top_k`.
Fetching wider than we return gives the fusion something to work with and feeds
the Phase 2 reranker a richer candidate pool.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25Retriever
from .config import Settings
from .embeddings import CohereEmbedder
from .vectorstore import ChromaStore, RetrievedChunk


@dataclass
class FusedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float          # the RRF score
    metadata: dict
    dense_rank: int | None = None   # 1-based rank in the vector list, if present
    lexical_rank: int | None = None  # 1-based rank in the BM25 list, if present


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    k: int,
    top_k: int,
) -> list[FusedChunk]:
    contributions: dict[str, dict] = {}

    def add(results: list[RetrievedChunk], which: str) -> None:
        for rank, chunk in enumerate(results, start=1):
            entry = contributions.setdefault(
                chunk.chunk_id,
                {"chunk": chunk, "rrf": 0.0, "dense_rank": None, "lexical_rank": None},
            )
            entry["rrf"] += 1.0 / (k + rank)
            entry[f"{which}_rank"] = rank

    add(dense, "dense")
    add(lexical, "lexical")

    fused = [
        FusedChunk(
            chunk_id=e["chunk"].chunk_id,
            doc_id=e["chunk"].doc_id,
            text=e["chunk"].text,
            score=e["rrf"],
            metadata=e["chunk"].metadata,
            dense_rank=e["dense_rank"],
            lexical_rank=e["lexical_rank"],
        )
        for e in contributions.values()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_k]


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        embedder: CohereEmbedder,
        bm25: BM25Retriever | None = None,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        # Build the BM25 index once from the same corpus in Chroma.
        self.bm25 = bm25 or BM25Retriever.from_store(store)

    def retrieve(
        self, query: str, top_k: int | None = None, fetch_k: int | None = None
    ) -> list[FusedChunk]:
        r = self.settings.retrieval
        top_k = top_k or r.top_k
        fetch_k = fetch_k or getattr(r, "fetch_k", max(top_k * 4, 20))
        rrf_k = getattr(r, "rrf_k", 60)

        query_vec = self.embedder.embed_query(query)
        dense = self.store.query(query_vec, top_k=fetch_k)
        lexical = self.bm25.retrieve(query, top_k=fetch_k)

        return reciprocal_rank_fusion(dense, lexical, k=rrf_k, top_k=top_k)
