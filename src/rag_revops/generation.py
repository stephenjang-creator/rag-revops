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
from .observability import count_llm_call, observe, record, record_usage
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
        # Client created lazily on first use so construction needs no key (tests
        # can build the object and stub the call); real calls require the key.
        self._client = None

    def _get_client(self):
        if self._client is None:
            secrets = load_secrets()
            if not secrets.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            self._client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
        return self._client

    @observe("generation", as_type="generation")
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        cfg = self.settings.generation
        prompts = self.settings.prompts
        decline_message = prompts.decline_message

        if not chunks:
            # No context to generate from: this is a decline, not a model call.
            record(no_context=True, declined=True)
            return AnswerResult(
                question=question,
                answer=decline_message,
                citations=[],
                declined=True,
                model=cfg.model,
            )

        context, citations = _format_context(chunks)
        user_msg = prompts.user_template.format(question=question, context=context)

        count_llm_call()
        resp = self._get_client().messages.create(
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

        # Trace the exact prompt sent, the response, tokens, and grounding.
        usage = getattr(resp, "usage", None)
        record_usage(
            model=cfg.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
        record(
            system_prompt=prompts.system,
            user_prompt=user_msg,
            answer=answer,
            n_context_chunks=len(chunks),
            n_citations=len(used),
            declined=declined,
            uncited_answer=(not declined and not used),  # quality red flag
        )

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
