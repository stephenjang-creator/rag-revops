"""BM25 lexical retrieval.

Dense (vector) retrieval is strong on paraphrase and meaning but weak on exact
lexical hits — defined terms ("Confidential Information"), section references
("Section 7.2"), dollar/day figures ("Net 30", "$50,000"), and rare tokens that
embeddings smear together. BM25 keyword scoring complements it, which is why the
two get fused (see hybrid.py).

The index is built over the *same* chunks stored in Chroma (via store.get_all),
so the lexical and dense views never drift apart. It's rebuilt in-memory at
startup — fine for corpora up to the low-hundreds-of-thousands of chunks. Beyond
that you'd persist the index; noted as a future step.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .vectorstore import ChromaStore, RetrievedChunk

# Keep alphanumerics and a few legal-meaningful characters; split on the rest.
# We deliberately preserve tokens like "30" and "7.2" rather than stripping
# digits, since numeric terms carry real meaning in contracts.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        if not chunks:
            raise ValueError("BM25Retriever needs a non-empty chunk list.")
        self._chunks = chunks
        self._corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    @classmethod
    def from_store(cls, store: ChromaStore) -> "BM25Retriever":
        return cls(store.get_all())

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(tokenize(query))
        # Rank indices by score, descending, take top_k.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for idx in ranked[:top_k]:
            c = self._chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    text=c.text,
                    score=float(scores[idx]),
                    metadata=c.metadata,
                )
            )
        return out

    @property
    def size(self) -> int:
        return len(self._chunks)
