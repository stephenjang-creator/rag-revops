"""Clause drafting — suggest new clause language grounded in retrieved examples.

ClauseFinder does the precision retrieval (reranker-ranked real passages from the
corpus). This step turns those examples into a DRAFT: Claude synthesizes clean,
reusable clause language for the requested concept, grounded only in the retrieved
passages and citing which ones it drew from with [n] markers.

So the headline output is suggested phrasing for a *new* contract, and the
retrieved contracts become the citations behind it — the drafted language is
traceable to real agreements rather than invented. The model is told to decline
(a fixed message) when the examples don't actually cover the requested clause, so
it doesn't fabricate language the corpus can't support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic

from .clause_finder import ClauseOption
from .config import Settings, load_secrets
from .observability import count_llm_call, observe, record, record_usage

# Inline citation markers like "[1]" or "[2, 3]", with any leading space/tab so we
# don't leave a double space behind when we strip them for the clean copy.
_CITATION_MARKER = re.compile(r"[ \t]*\[\d+(?:\s*,\s*\d+)*\]")
# A parenthetical left holding only joiner words after the markers are removed,
# e.g. "([1] and [2])" -> "( and)". Cleared so the copy reads naturally.
_EMPTY_PARENS = re.compile(r"\(\s*(?:(?:and|or|&|/|,|;)\s*)*\)")


def strip_citations(text: str) -> str:
    """Return the draft with inline [n] citation markers removed, so it can be
    pasted straight into an order form. Also clears parentheticals that held only
    the markers (e.g. "([1] and [2])") and collapses the whitespace left behind."""
    out = _CITATION_MARKER.sub("", text)
    out = _EMPTY_PARENS.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    # Tidy any now-orphaned space before sentence punctuation.
    out = re.sub(r"[ \t]+([,.;:)])", r"\1", out)
    return out.strip()


def decline_reason(draft: str) -> str:
    """For a declined draft, the model's explanation after the decline sentence
    (empty if it gave none). Citations are left intact so any [n] resolves against
    the passages shown alongside it."""
    body = draft.lstrip()
    if body.startswith(DECLINE_MESSAGE):
        body = body[len(DECLINE_MESSAGE):]
    return body.strip()

# Emitted verbatim by the model when the retrieved examples don't support a draft,
# and detected here to flag the decline — kept in one place so prompt and
# detection can't drift.
DECLINE_MESSAGE = (
    "I couldn't find example language in the corpus to ground a suggestion for "
    "that clause."
)


@dataclass
class DraftedClause:
    query: str
    draft: str                      # suggested clause language, with inline [n] markers
    citations: list[ClauseOption]   # the examples the draft actually cited ([n])
    options: list[ClauseOption]     # all retrieved examples (marker i -> options[i-1])
    declined: bool
    model: str


class ClauseDrafter:
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

    @observe("clause_draft", as_type="generation")
    def draft(self, query: str, options: list[ClauseOption]) -> DraftedClause:
        cfg = self.settings.generation
        model = cfg.model

        if not options:
            # No retrieved language to ground a draft: decline without a model call.
            record(no_context=True, declined=True)
            return DraftedClause(
                query=query, draft=DECLINE_MESSAGE, citations=[], options=[],
                declined=True, model=model,
            )

        context = self._format_options(options)
        system = (
            "You are a contract-drafting assistant. Using ONLY the example clause "
            "language retrieved from real contracts below, draft one clean, reusable "
            "clause for the requested concept that a Deal Desk could drop into a new "
            "agreement.\n\n"
            "Rules:\n"
            "- Ground every part of the draft in the examples. Do not invent "
            "obligations, numbers, or terms that no example supports.\n"
            "- Cite the examples you drew from inline with their bracketed ids, e.g. "
            "[1], [3], right where you use them.\n"
            "- Prefer clear, modern plain-drafting language; you may generalize party "
            "names to neutral terms like 'either party' or 'the Receiving Party'.\n"
            "- Where the examples differ on a material term (e.g. a notice period), "
            "note the range or leave a clearly-marked [bracketed placeholder] rather "
            "than picking one arbitrarily.\n"
            f"- If the examples do not actually cover the requested clause, do NOT "
            f"draft one. Begin your reply with exactly this sentence: "
            f'"{DECLINE_MESSAGE}" and then, on a new line, briefly explain why (you '
            f"may cite [n]).\n"
            "Otherwise, output the clause text only, with inline [n] citations — no "
            "preamble."
        )
        user = (
            f"REQUESTED CLAUSE: {query}\n\n"
            f"EXAMPLE LANGUAGE FROM REAL CONTRACTS:\n{context}\n\n"
            "Draft the clause now, citing the examples you use."
        )

        count_llm_call()
        resp = self._get_client().messages.create(
            model=model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        draft = "".join(b.text for b in resp.content if b.type == "text").strip()

        # The model may emit the decline sentence followed by an explanation, so
        # match on prefix, not exact equality — otherwise a decline gets rendered
        # as if it were a clause (copy block and all).
        declined = draft.lstrip().startswith(DECLINE_MESSAGE)
        # Keep only the examples the draft actually referenced inline as [n].
        cited = (
            []
            if declined
            else [o for i, o in enumerate(options, start=1) if f"[{i}]" in draft]
        )

        usage = getattr(resp, "usage", None)
        record_usage(
            model=model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
        record(
            n_options=len(options),
            n_citations=len(cited),
            declined=declined,
            uncited_draft=(not declined and not cited),  # quality red flag
        )

        return DraftedClause(
            query=query, draft=draft, citations=cited, options=options,
            declined=declined, model=model,
        )

    @staticmethod
    def _format_options(options: list[ClauseOption]) -> str:
        blocks: list[str] = []
        for i, o in enumerate(options, start=1):
            blocks.append(f"[{i}] (source: {o.source_path})\n{o.text}")
        return "\n\n".join(blocks)
