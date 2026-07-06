"""Custom LLM-as-judge evaluation — no Ragas, no dependency conflicts.

Ragas is structurally incompatible with a langchain-1.x / LangGraph stack (it
eagerly imports old-langchain internals), so rather than fight the version war we
implement the three metrics directly against the existing Anthropic client. This
is also more transparent: each metric is a defined, inspectable procedure rather
than a black-box import.

Metrics (each in [0, 1], following the standard definitions):

  faithfulness       Decompose the answer into atomic claims; check each against
                     the retrieved context; score = supported_claims / total_claims.
                     This is the anti-hallucination metric the CI gate guards.

  answer_relevancy   How directly the answer addresses the question, judged on a
                     0..1 scale (irrelevant / partial / fully on-point).

  context_precision  Of the retrieved chunks, the fraction that are actually
                     relevant to answering the question — rewards signal over noise.

All three call Claude with temperature 0 and parse strict JSON. Each function is
defensive about malformed judge output: on a parse failure it returns None for
that item, and the aggregator ignores None rather than scoring it 0 (a judge
hiccup shouldn't look like a faithfulness failure).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

from rag_revops.config import Settings, load_secrets

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ItemScores:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None


def _parse_json(text: str) -> dict | None:
    """Extract the first JSON object from the model's reply, tolerantly."""
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class ClaudeJudge:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Client is created lazily on first use, not here — so tests can construct
        # the judge (and stub _ask_json) without a key; only real API calls require
        # ANTHROPIC_API_KEY.
        self._client = None
        self._model = settings.eval.judge_model
        self._max_tokens = settings.eval.judge_max_tokens

    def _get_client(self):
        if self._client is None:
            secrets = load_secrets()
            if not secrets.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            self._client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
        return self._client

    def _ask_json(self, system: str, user: str) -> dict | None:
        resp = self._get_client().messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json(text)

    # --- faithfulness ------------------------------------------------------
    def faithfulness(self, answer: str, contexts: list[str]) -> float | None:
        if not answer.strip() or not contexts:
            return None
        context_block = "\n\n---\n\n".join(
            f"[{i}] {c}" for i, c in enumerate(contexts, start=1)
        )
        system = (
            "You are a strict evaluator measuring FAITHFULNESS of an answer to its "
            "source context. Decompose the answer into atomic factual claims. For "
            "each claim, decide if it is SUPPORTED by the context (directly stated "
            "or clearly entailed) or NOT SUPPORTED. Ignore subjective phrasing; "
            "judge only factual content. Respond with strict JSON only."
        )
        user = (
            f"CONTEXT:\n{context_block}\n\n"
            f"ANSWER:\n{answer}\n\n"
            'Return JSON exactly of the form:\n'
            '{"claims": [{"claim": "...", "supported": true}], '
            '"n_claims": <int>, "n_supported": <int>}\n'
            "If the answer contains no verifiable factual claims, return "
            '{"claims": [], "n_claims": 0, "n_supported": 0}.'
        )
        data = self._ask_json(system, user)
        if not data:
            return None
        n = data.get("n_claims")
        s = data.get("n_supported")
        if not isinstance(n, int) or not isinstance(s, int) or n <= 0:
            # No verifiable claims -> faithfulness is not applicable; return None
            # so it's excluded rather than counted as a failure.
            return None
        return max(0.0, min(1.0, s / n))

    # --- answer relevancy --------------------------------------------------
    def answer_relevancy(self, question: str, answer: str) -> float | None:
        if not answer.strip():
            return None
        system = (
            "You evaluate ANSWER RELEVANCY: how directly an answer addresses the "
            "question asked, independent of whether it is correct. Respond with "
            "strict JSON only."
        )
        user = (
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
            'Return JSON exactly: {"relevancy": <float 0..1>, "reason": "..."}\n'
            "1.0 = fully addresses the question; 0.5 = partially/tangentially; "
            "0.0 = does not address it."
        )
        data = self._ask_json(system, user)
        if not data or "relevancy" not in data:
            return None
        try:
            return max(0.0, min(1.0, float(data["relevancy"])))
        except (TypeError, ValueError):
            return None

    # --- context precision -------------------------------------------------
    def context_precision(self, question: str, contexts: list[str]) -> float | None:
        if not contexts:
            return None
        context_block = "\n\n---\n\n".join(
            f"[{i}] {c}" for i, c in enumerate(contexts, start=1)
        )
        system = (
            "You evaluate CONTEXT PRECISION: of the retrieved passages, how many "
            "are actually relevant to answering the question. Judge each passage "
            "independently as relevant or not. Respond with strict JSON only."
        )
        user = (
            f"QUESTION:\n{question}\n\nPASSAGES:\n{context_block}\n\n"
            'Return JSON exactly: {"judgements": [true, false, ...], '
            '"n_relevant": <int>, "n_total": <int>}\n'
            "The judgements array must have one boolean per passage, in order."
        )
        data = self._ask_json(system, user)
        if not data:
            return None
        r = data.get("n_relevant")
        t = data.get("n_total")
        if not isinstance(r, int) or not isinstance(t, int) or t <= 0:
            return None
        return max(0.0, min(1.0, r / t))

    def score_item(
        self, question: str, answer: str, contexts: list[str], metrics: list[str]
    ) -> ItemScores:
        scores = ItemScores()
        if "faithfulness" in metrics:
            scores.faithfulness = self.faithfulness(answer, contexts)
        if "answer_relevancy" in metrics:
            scores.answer_relevancy = self.answer_relevancy(question, answer)
        if "context_precision" in metrics:
            scores.context_precision = self.context_precision(question, contexts)
        return scores
