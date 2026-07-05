"""LangGraph pipeline: retrieve -> (branch) -> generate | decline.

Using an explicit state graph pays off here: the "no relevant context" branch is
a real conditional edge rather than an `if` buried in a function, which makes the
human-in-the-loop / decline behavior legible and easy to extend in Phase 2
(insert rerank and citation-enforcement nodes between retrieve and generate).

    retrieve ──► route ──► generate ──► END
                   └────► decline ───► END
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import Settings, load_settings
from .embeddings import CohereEmbedder
from .generation import AnswerResult, Generator
from .hybrid import HybridRetriever
from .rerank import CohereReranker
from .retrieval import Retriever
from .vectorstore import ChromaStore, RetrievedChunk

# For pure DENSE retrieval with NO reranker, chunk scores are cosine similarity
# (~0..1), so we decline on a low best-match score. When the reranker is enabled,
# the decline decision moves to the rerank node, which owns a calibrated relevance
# score (see RerankConfig.min_relevance). RRF (hybrid) scores are not suitable for
# an absolute threshold, so hybrid-without-rerank routes on presence of results.
_MIN_DENSE_TOP_SCORE = 0.25


class PipelineState(TypedDict, total=False):
    question: str
    doc_id: str | None
    chunks: list[RetrievedChunk]
    result: AnswerResult


class RagPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.store = ChromaStore(self.settings.vectorstore)
        self.embedder = CohereEmbedder(self.settings.embeddings)
        self.use_hybrid = getattr(self.settings.retrieval, "use_hybrid", False)
        self.use_rerank = getattr(self.settings.rerank, "enabled", False)
        if self.use_hybrid:
            self.retriever = HybridRetriever(self.settings, self.store, self.embedder)
        else:
            self.retriever = Retriever(self.settings, self.store, self.embedder)
        self.reranker = CohereReranker(self.settings.rerank) if self.use_rerank else None
        self.generator = Generator(self.settings)
        self._graph = self._build()

    # --- nodes -------------------------------------------------------------
    def _retrieve(self, state: PipelineState) -> PipelineState:
        doc_id = state.get("doc_id")
        # HybridRetriever accepts a doc_id filter; the pure-dense fallback doesn't,
        # so only pass it when hybrid is active.
        if self.use_hybrid:
            chunks = self.retriever.retrieve(state["question"], doc_id=doc_id)
        else:
            chunks = self.retriever.retrieve(state["question"])
        return {"chunks": chunks}

    def _rerank(self, state: PipelineState) -> PipelineState:
        reranked = self.reranker.rerank(state["question"], state["chunks"])
        return {"chunks": reranked}

    def _generate(self, state: PipelineState) -> PipelineState:
        result = self.generator.generate(state["question"], state["chunks"])
        return {"result": result}

    def _decline(self, state: PipelineState) -> PipelineState:
        return {
            "result": AnswerResult(
                question=state["question"],
                answer=self.settings.prompts.decline_message,
                citations=[],
                declined=True,
                model=self.settings.generation.model,
            )
        }

    # --- routing -----------------------------------------------------------
    def _route_after_retrieve(self, state: PipelineState) -> str:
        """Route straight after retrieval (used when reranker is OFF)."""
        chunks = state.get("chunks") or []
        if not chunks:
            return "decline"
        if not self.use_hybrid and chunks[0].score < _MIN_DENSE_TOP_SCORE:
            return "decline"
        return "generate"

    def _route_after_rerank(self, state: PipelineState) -> str:
        """Route after reranking (used when reranker is ON). Decline decision is
        based on the calibrated rerank relevance score — this is the principled
        citation-enforcement gate."""
        chunks = state.get("chunks") or []
        if not chunks or chunks[0].score < self.settings.rerank.min_relevance:
            return "decline"
        return "generate"

    def _build(self):
        g = StateGraph(PipelineState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_node("decline", self._decline)
        g.set_entry_point("retrieve")

        if self.use_rerank:
            g.add_node("rerank", self._rerank)
            # retrieve always flows into rerank; the decline gate is post-rerank.
            g.add_edge("retrieve", "rerank")
            g.add_conditional_edges(
                "rerank",
                self._route_after_rerank,
                {"generate": "generate", "decline": "decline"},
            )
        else:
            g.add_conditional_edges(
                "retrieve",
                self._route_after_retrieve,
                {"generate": "generate", "decline": "decline"},
            )

        g.add_edge("generate", END)
        g.add_edge("decline", END)
        return g.compile()

    def ask(self, question: str, doc_id: str | None = None) -> AnswerResult:
        """Answer a question. If doc_id is given, retrieval is restricted to that
        single contract (used by the UI's 'ask about one contract' mode); otherwise
        it searches the whole corpus."""
        final = self._graph.invoke({"question": question, "doc_id": doc_id})
        return final["result"]

    def ask_with_contexts(self, question: str) -> tuple[AnswerResult, list[str]]:
        """Like ask(), but also returns the full text of the chunks that were in
        context at generation time. The eval needs the complete chunk texts (not
        the truncated citation snippets) to score whether the answer's claims are
        grounded. On a declined answer the contexts are whatever was retrieved."""
        final = self._graph.invoke({"question": question})
        chunks = final.get("chunks") or []
        contexts = [c.text for c in chunks]
        return final["result"], contexts
