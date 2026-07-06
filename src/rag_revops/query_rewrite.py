"""Query rewriting — reformulate a user's raw question into an ideal retrieval query.

Deal Desk users type shorthand, abbreviations, and jargon: "can either party do a
TFC", "whats the MFN", "IP ownership?". These embed poorly against contract language
(the surface-form gap that hurts retrieval). A small LLM pass rewrites the input into
a well-formed, expanded question that retrieval handles well — expanding abbreviations
(TFC -> termination for convenience), clarifying intent, and phrasing it the way a
contract would.

The original is always preserved so the UI can show the user what the system
interpreted ("Interpreted 'TFC' as ...") — transparency matters for trust.

One cheap Claude call. Reused by both single-doc and analytical modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from .config import Settings, load_secrets
from .observability import count_llm_call, observe, record, record_usage


@dataclass
class RewrittenQuery:
    original: str
    rewritten: str
    changed: bool  # False when the rewrite is materially the same as the original


class QueryRewriter:
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

    @observe("query_rewrite", as_type="generation")
    def rewrite(self, question: str) -> RewrittenQuery:
        q = question.strip()
        if not q:
            return RewrittenQuery(original=question, rewritten=question, changed=False)

        cfg = self.settings.generation
        system = (
            "You reformulate a user's question about legal/commercial contracts into "
            "a clear, well-formed retrieval query. Users type shorthand, abbreviations, "
            "and jargon; your job is to expand and clarify so a document search finds "
            "the right clauses.\n\n"
            "Rules:\n"
            "- Expand contract abbreviations: TFC = termination for convenience, "
            "MFN = most favored nation, IP = intellectual property, SLA = service "
            "level agreement, NDA = confidentiality/non-disclosure, ROFR = right of "
            "first refusal, etc.\n"
            "- Phrase it the way a contract would express the concept, not as jargon.\n"
            "- Keep the user's intent; do not invent new criteria they didn't ask about.\n"
            "- Return ONLY the rewritten question, one line, no preamble or quotes."
        )
        count_llm_call()
        resp = self._get_client().messages.create(
            model=cfg.model,
            max_tokens=200,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": q}],
        )
        rewritten = "".join(b.text for b in resp.content if b.type == "text").strip()

        usage = getattr(resp, "usage", None)
        record_usage(
            model=cfg.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

        # Guard against an empty or degenerate rewrite — fall back to the original.
        if not rewritten:
            record(original=q, rewritten=q, changed=False)
            return RewrittenQuery(original=q, rewritten=q, changed=False)

        changed = _normalize(rewritten) != _normalize(q)
        record(original=q, rewritten=rewritten, changed=changed)
        return RewrittenQuery(original=q, rewritten=rewritten, changed=changed)


def _normalize(s: str) -> str:
    return " ".join(s.lower().split()).rstrip("?.")
