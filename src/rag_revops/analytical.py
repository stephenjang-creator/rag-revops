"""Analytical (cross-corpus) retrieval — contract-level.

The single-doc retriever answers "what does THIS contract say about X?" by pulling
the top-k most relevant chunks overall. Analytical queries are different — "which
contracts have termination for cause?" needs *coverage*: every contract that matches
must get a fair chance to surface, not just the contracts that happen to own the
globally top-ranked chunks.

Chunk-level top-k retrieval can't do this. A clause spread across 40% of the corpus
produces relevant chunks in many contracts, but a fixed chunk budget (e.g. 150 of
1,500) only covers ~10% of them — so most matching contracts are never even fetched.
Recall collapses regardless of thresholds.

The fix is CONTRACT-LEVEL retrieval:
  1. Fetch a wide dense pool (wide enough to touch every contract at least once).
  2. Collapse to ONE representative chunk per contract — its best dense hit.
  3. Rerank those representatives (one per contract) with the cross-encoder.
  4. Return all contracts (ordered by rerank) as candidates; an LLM judge decides
     membership downstream, since the reranker can't recognize paraphrased clauses.

Now every contract is represented in the rerank step, so recall is bounded by the
reranker's judgement, not by a chunk budget. With ~100 contracts this is cheap —
we rerank ~100 chunks, one per contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .embeddings import CohereEmbedder
from .rerank import CohereReranker
from .vectorstore import ChromaStore, RetrievedChunk


@dataclass
class ContractMatch:
    doc_id: str
    source_path: str
    best_score: float                 # rerank score of this contract's representative
    # Representative dense-pool excerpts for this contract (best first). The LLM
    # judge reads their .text; they come from the dense pool, so RetrievedChunk.
    chunks: list[RetrievedChunk] = field(default_factory=list)


class AnalyticalRetriever:
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

    def retrieve(self, query: str) -> list[ContractMatch]:
        a = self.settings.analytical

        # 1. Wide dense fetch. We size the pool to comfortably exceed the number of
        #    chunks in the store so every contract is represented; Chroma caps at the
        #    collection size, so over-asking is safe.
        pool_size = min(a.pool_size, self.store.count())
        query_vec = self.embedder.embed_query(query)
        pool = self.store.query(query_vec, top_k=pool_size)
        if not pool:
            return []

        # 2. Collapse to a few representative chunks per contract — its best dense
        #    hits. The LLM judge needs enough context to spot a paraphrased clause,
        #    so we keep several chunks, not just the single best.
        by_contract: dict[str, list[RetrievedChunk]] = {}
        for ch in pool:  # pool is best-first
            bucket = by_contract.setdefault(ch.doc_id, [])
            if len(bucket) < a.chunks_per_contract:
                bucket.append(ch)
        # One representative (best chunk) per contract drives the rerank ordering.
        representatives = [chunks[0] for chunks in by_contract.values()]

        # 3. Rerank the representatives for ORDERING only. The reranker scores
        #    literal phrasings high but paraphrased legal clauses ("without cause",
        #    "sole discretion") near zero, so we do NOT use its score as a
        #    membership gate — that would drop most genuine matches. Instead the
        #    LLM judge (AnalyticalGenerator) decides membership with legal-
        #    equivalence knowledge the cross-encoder lacks. Rerank order just
        #    prioritizes which contracts the judge sees first.
        reranked = self.reranker.rerank(query, representatives, top_n=len(representatives))

        # 4. Return ALL candidates as matches (ordered by rerank). Membership is
        #    decided downstream by the LLM judge, not by a score threshold.
        matches: list[ContractMatch] = []
        for rc in reranked:
            # rc.doc_id is always a key in by_contract (representatives derive
            # from it), so the buckets are the RetrievedChunk excerpts we want.
            matches.append(
                ContractMatch(
                    doc_id=rc.doc_id,
                    source_path=str(rc.metadata.get("source_path", rc.doc_id)),
                    best_score=rc.score,
                    chunks=by_contract.get(rc.doc_id, []),
                )
            )

        return matches[: a.max_contracts]
