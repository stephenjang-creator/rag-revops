"""Clause language finder — the precision-retrieval showcase for the reranker.

Analytical mode (analytical.py) deliberately demotes the cross-encoder to
ordering-only, because it scores paraphrased clauses ("terminate without cause")
near zero and would drop genuine matches. This mode is the mirror image — the
reranker's home turf.

"Give me some language options for a termination-for-convenience clause" (or an
NDA, a limitation-of-liability, an indemnity…) is a PRECISION request: the user
wants the passages that most literally express the clause, to reuse as drafting
references. That is exactly what a cross-encoder is good at, so here the
calibrated rerank score is a real signal — we threshold on it, and a low score
genuinely means "not good example language for this clause".

Pipeline:
  1. Embed the (rewritten) clause description and pull a dense candidate pool.
  2. Rerank the pool with the cross-encoder — best-first, scored ~[0, 1].
  3. Keep the best passage PER CONTRACT, so the options are diverse examples from
     different agreements rather than three chunks of the same one.
  4. Return those above the relevance floor, best first. No LLM judge — this is
     retrieval + rerank by design.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embeddings import CohereEmbedder
from .observability import (
    RequestTimer,
    flush,
    observe,
    record,
    trace_context,
)
from .rerank import CohereReranker
from .vectorstore import ChromaStore


@dataclass
class ClauseOption:
    doc_id: str
    source_path: str
    text: str
    score: float          # cross-encoder relevance in ~[0, 1]
    chunk_id: str


@dataclass
class ClauseResult:
    query: str
    options: list[ClauseOption]
    n_considered: int     # passages the reranker scored


class ClauseFinder:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        embedder: CohereEmbedder,
        reranker: CohereReranker | None = None,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or CohereReranker(settings.rerank)

    @observe("clause_find")
    def find(self, query: str) -> ClauseResult:
        with (
            trace_context("clause_query", tags=["clause-language"], question=query,
                          config_version=self.settings.version),
            RequestTimer(mode="clause", question=query) as timer,
        ):
            result = self._find_inner(query)
            # A clause search with zero options above the floor is a decline-like
            # outcome — the corpus has no good example language for this clause.
            timer.set_result(
                declined=not result.options,
                n_citations=len(result.options),
                n_context_chunks=result.n_considered,
            )
        flush()
        return result

    def _find_inner(self, query: str) -> ClauseResult:
        c = self.settings.clause

        # Dense candidate pool. Sized to a few dozen; the reranker then does the
        # precision work. Chroma caps at the collection size, so over-asking is safe.
        pool_size = min(c.pool_size, self.store.count())
        if pool_size <= 0:
            return ClauseResult(query=query, options=[], n_considered=0)
        query_vec = self.embedder.embed_query(query)
        pool = self.store.query(query_vec, top_k=pool_size)
        if not pool:
            return ClauseResult(query=query, options=[], n_considered=0)

        # The cross-encoder scores each (query, passage) pair jointly. Ask for
        # scores on the whole pool (top_n=len) so the per-contract dedup below sees
        # every candidate, not just the reranker's default top slice.
        reranked = self.reranker.rerank(query, pool, top_n=len(pool))

        # Keep the best passage per contract, above the relevance floor. `reranked`
        # is best-first, so the first time we see a score below the floor, nothing
        # after it qualifies either — stop.
        seen: set[str] = set()
        options: list[ClauseOption] = []
        for rc in reranked:
            if rc.score < c.min_relevance:
                break
            if rc.doc_id in seen:
                continue
            seen.add(rc.doc_id)
            options.append(
                ClauseOption(
                    doc_id=rc.doc_id,
                    source_path=str(rc.metadata.get("source_path", rc.doc_id)),
                    text=rc.text,
                    score=rc.score,
                    chunk_id=rc.chunk_id,
                )
            )
            if len(options) >= c.max_options:
                break

        record(
            n_considered=len(reranked),
            n_options=len(options),
            top_score=(options[0].score if options else None),
            min_kept_score=(options[-1].score if options else None),
        )
        return ClauseResult(query=query, options=options, n_considered=len(reranked))
