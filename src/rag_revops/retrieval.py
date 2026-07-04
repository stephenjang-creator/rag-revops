"""Retrieval: embed the query and pull the top-k chunks.

Phase 1 is pure dense retrieval. Phase 2 will wrap this with BM25 + reciprocal
rank fusion and a Cohere cross-encoder rerank step — the interface here is kept
deliberately small so that swap is additive.
"""

from __future__ import annotations

from .config import Settings
from .embeddings import CohereEmbedder
from .vectorstore import ChromaStore, RetrievedChunk


class Retriever:
    def __init__(self, settings: Settings, store: ChromaStore, embedder: CohereEmbedder):
        self.settings = settings
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.settings.retrieval.top_k
        query_vec = self.embedder.embed_query(query)
        return self.store.query(query_vec, top_k=k)
