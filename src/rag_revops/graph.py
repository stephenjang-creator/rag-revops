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
from .generation import DECLINE_SENTINEL, AnswerResult, Generator
from .retrieval import Retriever
from .vectorstore import ChromaStore, RetrievedChunk

# Below this best-match similarity, we treat retrieval as a miss and decline
# rather than feeding weak context to the generator. Tune in Phase 3 against eval.
_MIN_TOP_SCORE = 0.25


class PipelineState(TypedDict, total=False):
    question: str
    chunks: list[RetrievedChunk]
    result: AnswerResult


class RagPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.store = ChromaStore(self.settings.vectorstore)
        self.embedder = CohereEmbedder(self.settings.embeddings)
        self.retriever = Retriever(self.settings, self.store, self.embedder)
        self.generator = Generator(self.settings)
        self._graph = self._build()

    # --- nodes -------------------------------------------------------------
    def _retrieve(self, state: PipelineState) -> PipelineState:
        chunks = self.retriever.retrieve(state["question"])
        return {"chunks": chunks}

    def _generate(self, state: PipelineState) -> PipelineState:
        result = self.generator.generate(state["question"], state["chunks"])
        return {"result": result}

    def _decline(self, state: PipelineState) -> PipelineState:
        return {
            "result": AnswerResult(
                question=state["question"],
                answer=DECLINE_SENTINEL,
                citations=[],
                declined=True,
                model=self.settings.generation.model,
            )
        }

    # --- routing -----------------------------------------------------------
    @staticmethod
    def _route(state: PipelineState) -> str:
        chunks = state.get("chunks") or []
        if not chunks or chunks[0].score < _MIN_TOP_SCORE:
            return "decline"
        return "generate"

    def _build(self):
        g = StateGraph(PipelineState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_node("decline", self._decline)
        g.set_entry_point("retrieve")
        g.add_conditional_edges(
            "retrieve", self._route, {"generate": "generate", "decline": "decline"}
        )
        g.add_edge("generate", END)
        g.add_edge("decline", END)
        return g.compile()

    def ask(self, question: str) -> AnswerResult:
        final = self._graph.invoke({"question": question})
        return final["result"]
