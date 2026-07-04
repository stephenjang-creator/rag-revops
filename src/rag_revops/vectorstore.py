"""ChromaDB persistence and query.

We pass embeddings in explicitly (computed by Cohere) rather than letting Chroma
manage an embedding function, so the embedding model stays a single, versioned
choice in config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking import Chunk
from .config import VectorStoreConfig


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float          # similarity (1 - cosine distance)
    metadata: dict


class ChromaStore:
    def __init__(self, cfg: VectorStoreConfig):
        self.cfg = cfg
        Path(cfg.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=cfg.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=cfg.collection,
            metadata={"hnsw:space": cfg.distance},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": c.doc_id, "index": c.index, **c.metadata} for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", ""),
                    text=text,
                    score=1.0 - float(dist),
                    metadata=meta,
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def get_all(self) -> list[RetrievedChunk]:
        """Return every stored chunk (no scores). Used to build the BM25 index
        over the same corpus, so dense and lexical retrieval never drift apart."""
        res = self._collection.get(include=["documents", "metadatas"])
        out: list[RetrievedChunk] = []
        for cid, text, meta in zip(
            res["ids"], res["documents"], res["metadatas"], strict=True
        ):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", ""),
                    text=text,
                    score=0.0,
                    metadata=meta,
                )
            )
        return out
