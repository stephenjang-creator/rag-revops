"""ChromaDB persistence and query.

We pass embeddings in explicitly (computed by Cohere) rather than letting Chroma
manage an embedding function, so the embedding model stays a single, versioned
choice in config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking import Chunk
from .config import VectorStoreConfig


class ChunkLike(Protocol):
    """Structural type for anything that can flow through the pipeline as a
    retrieved passage. RetrievedChunk, FusedChunk (hybrid), and RerankedChunk
    all satisfy it, so stages can accept "a chunk" without caring which concrete
    class produced it — they only read these fields.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict


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
            # Chroma's stubs type embeddings as numpy-oriented; a list[list[float]]
            # is accepted at runtime. Cast to keep the public API list-based.
            embeddings=cast(Any, embeddings),
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": c.doc_id, "index": c.index, **c.metadata} for c in chunks
            ],
        )

    def query(
        self, query_embedding: list[float], top_k: int, doc_id: str | None = None
    ) -> list[RetrievedChunk]:
        # Optionally restrict retrieval to a single contract via metadata filter.
        # This is what lets single-doc mode answer about ONE contract instead of
        # blending passages from many.
        where = {"doc_id": doc_id} if doc_id else None
        res = self._collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
            where=cast(Any, where),
            include=cast(Any, ["documents", "metadatas", "distances"]),
        )
        # Chroma types every result field as Optional; on a successful query the
        # requested keys are present. Guard defensively and default to empty.
        ids_all = res.get("ids") or [[]]
        docs_all = res.get("documents") or [[]]
        metas_all = res.get("metadatas") or [[]]
        dists_all = res.get("distances") or [[]]
        ids = ids_all[0]
        docs = docs_all[0]
        metas = metas_all[0]
        dists = dists_all[0]

        out: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
            meta_dict = dict(meta) if meta else {}
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=str(meta_dict.get("doc_id", "")),
                    text=text or "",
                    score=1.0 - float(dist),
                    metadata=meta_dict,
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def list_contracts(self) -> list[str]:
        """Return the sorted distinct contract (doc_id) list, for a UI picker."""
        res = self._collection.get(include=cast(Any, ["metadatas"]))
        metas = res.get("metadatas") or []
        ids = {
            str(m.get("doc_id", ""))
            for m in metas
            if m and m.get("doc_id")
        }
        return sorted(ids)

    def get_all(self) -> list[RetrievedChunk]:
        """Return every stored chunk (no scores). Used to build the BM25 index
        over the same corpus, so dense and lexical retrieval never drift apart."""
        res = self._collection.get(include=cast(Any, ["documents", "metadatas"]))
        ids_all = res.get("ids") or []
        docs_all = res.get("documents") or []
        metas_all = res.get("metadatas") or []
        out: list[RetrievedChunk] = []
        for cid, text, meta in zip(ids_all, docs_all, metas_all, strict=True):
            meta_dict = dict(meta) if meta else {}
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=str(meta_dict.get("doc_id", "")),
                    text=text or "",
                    score=0.0,
                    metadata=meta_dict,
                )
            )
        return out
