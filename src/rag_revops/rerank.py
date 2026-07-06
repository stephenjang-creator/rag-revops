"""Cohere cross-encoder reranking.

Fusion (hybrid.py) ranks candidates by *position* across two retrievers, which is
a good recall net but a weak precision signal — it never actually reads the query
against each chunk. The reranker does exactly that: Cohere's rerank endpoint is a
cross-encoder that scores each (query, chunk) pair jointly, producing a calibrated
relevance score in ~[0, 1].

That gives us two things:
  1. Sharper ordering — the chunk that truly answers the question rises, even if
     it ranked mid-pack after fusion.
  2. A *stable, absolute* relevance score we can threshold on. RRF scores couldn't
     support an absolute decline cutoff; rerank scores can. So this is also what
     makes principled citation-enforcement / "decline rather than guess" work.

Input is any object with `.text` (FusedChunk or RetrievedChunk). Output is a list
of RerankedChunk, sorted best-first, each carrying the original chunk plus its
relevance score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import cohere

from .config import RerankConfig, load_secrets
from .observability import observe, record


class _HasText(Protocol):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


@dataclass
class RerankedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float          # Cohere relevance score in ~[0, 1]
    metadata: dict


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


class CohereReranker:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        # Client is created lazily on first use, not here — so tests can construct
        # the reranker and inject a stub client without a key, and only real API
        # calls require COHERE_API_KEY.
        self._client = None

    def _get_client(self):
        if self._client is None:
            secrets = load_secrets()
            if not secrets.cohere_api_key:
                raise RuntimeError("COHERE_API_KEY is not set (see .env.example).")
            self._client = cohere.Client(api_key=secrets.cohere_api_key)
        return self._client

    def _rerank_with_retry(self, query: str, documents: list[str], top_n: int):
        client = self._get_client()
        attempt = 0
        while True:
            try:
                return client.rerank(
                    query=query,
                    documents=documents,
                    model=self.cfg.model,
                    top_n=min(top_n, len(documents)),
                )
            except Exception as exc:  # noqa: BLE001 - retry rate limits, else raise
                if not _is_rate_limit(exc) or attempt >= self.cfg.max_retries:
                    raise
                wait = min(self.cfg.backoff_base_s * (2 ** attempt), self.cfg.backoff_cap_s)
                attempt += 1
                print(
                    f"  [rate limit] rerank waiting {wait:.0f}s then retrying "
                    f"(attempt {attempt}/{self.cfg.max_retries})…"
                )
                time.sleep(wait)

    @observe("rerank")
    def rerank(
        self, query: str, chunks: list[_HasText], top_n: int | None = None
    ) -> list[RerankedChunk]:
        if not chunks:
            record(candidates_in=0, candidates_out=0, dropped=0)
            return []
        # Default to the single-doc top_n cap; callers (e.g. analytical mode) can
        # override to get scores for ALL chunks by passing top_n=len(chunks).
        effective_top_n = top_n if top_n is not None else self.cfg.top_n
        documents = [c.text for c in chunks]
        resp = self._rerank_with_retry(query, documents, effective_top_n)

        out: list[RerankedChunk] = []
        for item in resp.results:
            src = chunks[item.index]
            out.append(
                RerankedChunk(
                    chunk_id=src.chunk_id,
                    doc_id=src.doc_id,
                    text=src.text,
                    score=float(item.relevance_score),
                    metadata=src.metadata,
                )
            )
        # Cohere returns results sorted by relevance, but sort defensively.
        out.sort(key=lambda c: c.score, reverse=True)

        # Trace how the reranker reordered the pool. `dropped` surfaces the
        # top_n result-cap that once silently discarded 95/100 contracts — now
        # a standing metric that would catch its own regression.
        record(
            candidates_in=len(chunks),
            top_n_requested=effective_top_n,
            candidates_out=len(out),
            dropped=len(chunks) - len(out),
            top_score=(out[0].score if out else None),
            min_kept_score=(out[-1].score if out else None),
            reordered_ids=[c.chunk_id for c in out],
        )
        return out
