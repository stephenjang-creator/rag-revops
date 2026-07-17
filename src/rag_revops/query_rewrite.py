"""Query rewriting — reformulate a user's raw question into an ideal retrieval query.

Deal Desk users type shorthand, abbreviations, and jargon: "can either party do a
TFC", "whats the MFN", "IP ownership?". These embed poorly against contract language
(the surface-form gap that hurts retrieval). A small LLM pass rewrites the input into
a well-formed, expanded question that retrieval handles well — expanding abbreviations
(TFC -> termination for convenience), clarifying intent, and phrasing it the way a
contract would.

The original is always preserved so the UI can show the user what the system
interpreted ("Interpreted 'TFC' as ...") — transparency matters for trust.

Provider-flexible: rewriting runs on EVERY query, so it defaults to Cohere's free
`command` model — reusing the Cohere key the app already collects for embeddings +
rerank, which keeps rewrite off the Anthropic bill with no extra key or dependency.
Falls back to Anthropic when no Cohere key is present. Selected via
`settings.rewrite.provider`.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
import cohere

from .config import RewriteConfig, Settings, load_secrets
from .observability import count_llm_call, observe, record, record_usage


@dataclass
class RewrittenQuery:
    original: str
    rewritten: str
    changed: bool  # False when the rewrite is materially the same as the original


# Shared across providers so the interpretation is identical no matter who runs it.
_SYSTEM = (
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


class QueryRewriter:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Clients are created lazily on first use so construction needs no key
        # (tests can build the object and inject a stub); real calls require it.
        self._anthropic = None
        self._cohere = None

    def _get_anthropic(self):
        if self._anthropic is None:
            secrets = load_secrets()
            if not secrets.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            self._anthropic = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
        return self._anthropic

    def _get_cohere(self):
        if self._cohere is None:
            secrets = load_secrets()
            if not secrets.cohere_api_key:
                raise RuntimeError("COHERE_API_KEY is not set (see .env.example).")
            self._cohere = cohere.Client(api_key=secrets.cohere_api_key)
        return self._cohere

    @staticmethod
    def _effective_provider(preferred: str) -> str:
        """Resolve which provider to actually use given the configured preference
        and which keys are present. A missing Cohere key doesn't break the app — it
        just rewrites via Anthropic; with no key at all, rewriting is skipped."""
        secrets = load_secrets()
        has_cohere = bool(secrets.cohere_api_key)
        has_anthropic = bool(secrets.anthropic_api_key)
        if preferred == "cohere" and has_cohere:
            return "cohere"
        if preferred == "anthropic" and has_anthropic:
            return "anthropic"
        # Fallbacks: prefer whichever key is available.
        if has_cohere:
            return "cohere"
        if has_anthropic:
            return "anthropic"
        return "none"

    @observe("query_rewrite", as_type="generation")
    def rewrite(self, question: str) -> RewrittenQuery:
        q = question.strip()
        if not q:
            return RewrittenQuery(original=question, rewritten=question, changed=False)

        cfg = self.settings.rewrite
        provider = self._effective_provider(cfg.provider)

        if provider == "cohere":
            rewritten = self._rewrite_cohere(q, cfg)
        elif provider == "anthropic":
            rewritten = self._rewrite_anthropic(q, cfg)
        else:
            # No usable key — skip rewriting rather than erroring; retrieval still
            # runs on the raw question.
            record(original=q, rewritten=q, changed=False, provider="none")
            return RewrittenQuery(original=q, rewritten=q, changed=False)

        # Guard against an empty or degenerate rewrite — fall back to the original.
        if not rewritten:
            record(original=q, rewritten=q, changed=False, provider=provider)
            return RewrittenQuery(original=q, rewritten=q, changed=False)

        changed = _normalize(rewritten) != _normalize(q)
        record(original=q, rewritten=rewritten, changed=changed, provider=provider)
        return RewrittenQuery(original=q, rewritten=rewritten, changed=changed)

    def _rewrite_anthropic(self, q: str, cfg: RewriteConfig) -> str:
        count_llm_call()
        resp = self._get_anthropic().messages.create(
            model=cfg.anthropic_model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=_SYSTEM,
            messages=[{"role": "user", "content": q}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        usage = getattr(resp, "usage", None)
        record_usage(
            model=cfg.anthropic_model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
        return text

    def _rewrite_cohere(self, q: str, cfg: RewriteConfig) -> str:
        count_llm_call()
        # Cohere v1 chat: `preamble` is the system instruction, `message` the input.
        resp = self._get_cohere().chat(
            model=cfg.cohere_model,
            message=q,
            preamble=_SYSTEM,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        text = (getattr(resp, "text", "") or "").strip()
        record_usage(
            model=cfg.cohere_model,
            input_tokens=_cohere_tokens(resp, "input_tokens"),
            output_tokens=_cohere_tokens(resp, "output_tokens"),
        )
        return text


def _cohere_tokens(resp, field: str) -> int | None:
    """Best-effort token count from a Cohere chat response. The token block has
    lived under a couple of names across SDK versions; try each, default to None."""
    meta = getattr(resp, "meta", None)
    if meta is None:
        return None
    for holder in ("tokens", "billed_units"):
        obj = getattr(meta, holder, None)
        val = getattr(obj, field, None) if obj is not None else None
        if val is not None:
            return int(val)
    return None


def _normalize(s: str) -> str:
    return " ".join(s.lower().split()).rstrip("?.")
