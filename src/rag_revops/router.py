"""Skill router — pick the right skill for a Deal Desk question.

The app has three skills over the same corpus:

  • ask_one_contract  — answer about ONE named contract   (graph.RagPipeline.ask)
  • find_contracts    — which contracts across the book have clause X
                        (analytical.AnalyticalRetriever + AnalyticalGenerator)
  • draft_clause      — draft NEW reusable clause language for an order form
                        (clause_finder.ClauseFinder + clause_drafting.ClauseDrafter)

Instead of making the user pick one from a menu, this router reads the question
and classifies it. It's a single cheap, structured LLM call (forced tool use, so
the output is a validated object, not free text) plus a small amount of
deterministic post-processing. When the intent is genuinely ambiguous it sets
`needs_clarification` rather than guessing — consistent with the rest of the
app's decline-rather-than-guess design.

Classification only. Executing the chosen skill stays with the caller (the
Streamlit UI), because each skill returns a different result type and renders
differently — a shared graph state would be an awkward fit.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import anthropic

from .config import Settings, load_secrets
from .observability import count_llm_call, observe, record, record_usage

# Skill identifiers — kept as constants so the UI and tests reference one source.
SKILL_ASK_ONE = "ask_one_contract"
SKILL_FIND_MANY = "find_contracts"
SKILL_DRAFT = "draft_clause"
_SKILLS = (SKILL_ASK_ONE, SKILL_FIND_MANY, SKILL_DRAFT)


@dataclass
class RouterDecision:
    skill: str                       # one of _SKILLS
    contract_hint: str | None        # contract/party the user named (ask_one_contract)
    needs_clarification: bool        # too ambiguous to route confidently
    reason: str                      # one-sentence rationale, shown to the user


_ROUTE_TOOL = {
    "name": "route",
    "description": "Route the Deal Desk user's request to exactly one skill.",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "enum": list(_SKILLS),
                "description": (
                    "ask_one_contract: they ask about a SPECIFIC named contract. "
                    "find_contracts: they ask WHICH contracts across the book have "
                    "some clause. draft_clause: they want NEW reusable language to "
                    "put in an order form or new contract."
                ),
            },
            "contract_hint": {
                "type": "string",
                "description": (
                    "For ask_one_contract, the contract name or party the user "
                    "named, verbatim. Empty string if none was named."
                ),
            },
            "needs_clarification": {
                "type": "boolean",
                "description": "True if it's genuinely ambiguous which skill applies.",
            },
            "reason": {"type": "string", "description": "One sentence explaining the choice."},
        },
        "required": ["skill", "needs_clarification", "reason"],
        "additionalProperties": False,
    },
}

_SYSTEM = (
    "You route a Deal Desk user's request to exactly one skill over a corpus of "
    "contracts. Judge by intent, not keywords.\n\n"
    "- ask_one_contract: the user asks about a SPECIFIC, named contract or party "
    "('what does the Acme MSA say about liability?', 'does the Pizza Fusion "
    "agreement allow assignment?'). Put the contract/party they named in "
    "contract_hint.\n"
    "- find_contracts: the user asks WHICH contracts across the whole book have a "
    "clause or property ('which agreements cap liability?', 'who has audit "
    "rights?', 'find contracts that allow termination for convenience').\n"
    "- draft_clause: the user wants NEW, reusable clause language to drop into an "
    "order form or a new contract ('give me a termination-for-convenience clause', "
    "'draft a mutual NDA', 'language options for an indemnity').\n\n"
    "The single-contract vs cross-corpus distinction is by intent: 'termination "
    "for convenience in the Acme MSA' is ask_one_contract; 'which contracts allow "
    "termination for convenience' is find_contracts. If you genuinely cannot tell "
    "which skill applies, set needs_clarification=true."
)


class QueryRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Client created lazily on first use so construction needs no key (tests
        # can build the object and inject a stub client); real calls require it.
        self._client = None

    def _get_client(self):
        if self._client is None:
            secrets = load_secrets()
            if not secrets.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            self._client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
        return self._client

    @observe("route", as_type="generation")
    def route(self, question: str) -> RouterDecision:
        cfg = self.settings.generation
        count_llm_call()
        resp = self._get_client().messages.create(
            model=cfg.model,
            max_tokens=400,
            temperature=cfg.temperature,
            system=_SYSTEM,
            tools=[_ROUTE_TOOL],
            tool_choice={"type": "tool", "name": "route"},
            messages=[{"role": "user", "content": question}],
        )

        data: dict = {}
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "route":
                data = dict(block.input)
                break

        usage = getattr(resp, "usage", None)
        record_usage(
            model=cfg.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

        skill = data.get("skill")
        # Unknown / missing skill → treat as ambiguous rather than mis-dispatching.
        if skill not in _SKILLS:
            decision = RouterDecision(
                skill=SKILL_ASK_ONE,
                contract_hint=None,
                needs_clarification=True,
                reason=str(data.get("reason", "Couldn't classify the request.")),
            )
        else:
            hint = (data.get("contract_hint") or "").strip() or None
            decision = RouterDecision(
                skill=skill,
                contract_hint=hint,
                needs_clarification=bool(data.get("needs_clarification", False)),
                reason=str(data.get("reason", "")).strip(),
            )
        record(
            skill=decision.skill,
            contract_hint=decision.contract_hint,
            needs_clarification=decision.needs_clarification,
        )
        return decision


def resolve_contract(hint: str | None, contracts: list[str]) -> str | None:
    """Map a free-text contract hint ('the Pizza Fusion agreement') to a real
    doc_id from the store. Tries a normalized substring match first (handles the
    packed CUAD ids like '030__PfHospitalityGroupInc_..._Franchise_Agreement'),
    then falls back to fuzzy matching. Returns None when nothing matches well —
    the caller should then ask the user to pick."""
    if not hint or not contracts:
        return None

    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    h = _norm(hint)
    if h:
        for c in contracts:
            if h in _norm(c):
                return c

    matches = difflib.get_close_matches(hint.lower(), [c.lower() for c in contracts],
                                        n=1, cutoff=0.4)
    if matches:
        for c in contracts:
            if c.lower() == matches[0]:
                return c
    return None
