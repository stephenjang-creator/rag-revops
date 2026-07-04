"""Cited-answer generation using the Anthropic API.

The passages are numbered [1..n] and the model is instructed (via the versioned
prompt in config) to cite those ids inline and to decline when unsupported.
The `AnswerResult` carries the answer text alongside the exact sources used, so
the caller can render verifiable citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from .config import Settings, load_secrets
from .vectorstore import RetrievedChunk


@dataclass
class Citation:
    marker: int          # the [n] shown to the model
    chunk_id: str
    doc_id: str
    source_path: str
    score: float
    snippet: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation]
    declined: bool
    model: str


def _format_context(chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []
    for i, ch in enumerate(chunks, start=1):
        source = ch.metadata.get("source_path", ch.doc_id)
        lines.append(f"[{i}] (source: {source})\n{ch.text}")
        citations.append(
            Citation(
                marker=i,
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                source_path=source,
                score=ch.score,
                snippet=ch.text[:240],
            )
        )
    return "\n\n".join(lines), citations


class Generator:
    def __init__(self, settings: Settings):
        self.settings = settings
        secrets = load_secrets()
        if not secrets.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
        self._client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        cfg = self.settings.generation
        prompts = self.settings.prompts
        decline_message = prompts.decline_message

        if not chunks:
            return AnswerResult(
                question=question,
                answer=decline_message,
                citations=[],
                declined=True,
                model=cfg.model,
            )

        context, citations = _format_context(chunks)
        user_msg = prompts.user_template.format(question=question, context=context)

        resp = self._client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=prompts.system,
            messages=[{"role": "user", "content": user_msg}],
        )
        answer = "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()

        declined = answer.strip() == decline_message
        used = [] if declined else _cited_only(answer, citations)

        return AnswerResult(
            question=question,
            answer=answer,
            citations=used,
            declined=declined,
            model=cfg.model,
        )


def _cited_only(answer: str, citations: list[Citation]) -> list[Citation]:
    """Keep only the citations the model actually referenced inline as [n]."""
    return [c for c in citations if f"[{c.marker}]" in answer]
