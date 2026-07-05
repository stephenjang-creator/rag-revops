"""Analytical answer generation — LLM-as-judge membership.

The cross-encoder reranker cannot recognize that a clause saying "terminate without
cause" is legally equivalent to "termination for convenience" — it scores paraphrased
clauses near zero (verified empirically). So membership in the answer set is decided
by an LLM judge (Claude), which HAS that legal-equivalence knowledge, rather than by
a rerank score threshold.

All candidate contracts are judged (batched, `judge_batch_size` per Claude call to
keep cost bounded). For each contract the judge sees a few excerpts and decides, with
explicit guidance that paraphrases count, whether the contract genuinely satisfies
the criterion — and must ground its reason in the excerpts (no guessing).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

from .analytical import ContractMatch
from .config import Settings, load_secrets

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ContractFinding:
    doc_id: str
    source_path: str
    reason: str
    score: float


@dataclass
class AnalyticalResult:
    question: str
    findings: list[ContractFinding]
    answer: str
    n_candidates: int
    model: str


def _format_batch(batch: list[ContractMatch]) -> str:
    blocks: list[str] = []
    for i, m in enumerate(batch, start=1):
        excerpts = "\n".join(f"      - {c.text[:700]}" for c in m.chunks)
        blocks.append(f"[{i}] contract_id: {m.doc_id}\n{excerpts}")
    return "\n\n".join(blocks)


class AnalyticalGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        secrets = load_secrets()
        if not secrets.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
        self._client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)

    def generate(self, question: str, matches: list[ContractMatch]) -> AnalyticalResult:
        cfg = self.settings.generation
        model = cfg.model

        if not matches:
            return AnalyticalResult(
                question=question,
                findings=[],
                answer="No contracts in the corpus match that criterion.",
                n_candidates=0,
                model=model,
            )

        batch_size = self.settings.analytical.judge_batch_size
        findings: list[ContractFinding] = []
        for start in range(0, len(matches), batch_size):
            batch = matches[start : start + batch_size]
            findings.extend(self._judge_batch(question, batch, model))

        answer = self._render_answer(findings)
        return AnalyticalResult(
            question=question,
            findings=findings,
            answer=answer,
            n_candidates=len(matches),
            model=model,
        )

    def _judge_batch(
        self, question: str, batch: list[ContractMatch], model: str
    ) -> list[ContractFinding]:
        cfg = self.settings.generation
        candidates = _format_batch(batch)
        system = (
            "You are PolicyLens analytical mode, judging which contracts satisfy a "
            "criterion. You receive numbered CANDIDATE contracts with excerpts. Your "
            "job is to CONFIRM every candidate whose excerpts satisfy the criterion — "
            "be thorough, not stingy; a genuine match that you miss is a failure.\n\n"
            "Judge by legal MEANING, not exact wording. A clause counts even if it "
            "never uses the query's exact words, as long as it expresses the concept. "
            "For 'termination for convenience', ALL of these count as matches:\n"
            "  - 'terminate ... without cause'\n"
            "  - 'terminate ... in its sole discretion'\n"
            "  - 'terminate ... for any reason'\n"
            "  - 'terminate ... without penalty'\n"
            "  - 'terminate ... at any time upon [N] days notice' (no cause required)\n"
            "  - 'may terminate for convenience'\n"
            "If an excerpt shows a party can end the agreement at will / without needing "
            "a reason or breach, CONFIRM it. Only drop candidates whose excerpts show no "
            "such right (e.g. termination only for cause/breach, or no termination "
            "language at all). Do not invent language that isn't there, but DO recognize "
            "equivalent phrasings. Respond with strict JSON only."
        )
        user = (
            f"CRITERION: {question}\n\n"
            f"CANDIDATES:\n{candidates}\n\n"
            'Return JSON exactly: {"matches": [{"n": <candidate number>, '
            '"reason": "<one sentence grounded in that candidate\'s excerpts>"}]}\n'
            "Include only candidates that genuinely satisfy the criterion by legal "
            'meaning. If none do, return {"matches": []}.'
        )
        resp = self._client.messages.create(
            model=model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return self._parse_findings(text, batch)

    @staticmethod
    def _parse_findings(text: str, batch: list[ContractMatch]) -> list[ContractFinding]:
        m = _JSON.search(text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        findings: list[ContractFinding] = []
        for entry in data.get("matches", []):
            n = entry.get("n")
            if not isinstance(n, int) or not (1 <= n <= len(batch)):
                continue
            src = batch[n - 1]
            findings.append(
                ContractFinding(
                    doc_id=src.doc_id,
                    source_path=src.source_path,
                    reason=str(entry.get("reason", "")).strip(),
                    score=src.best_score,
                )
            )
        return findings

    @staticmethod
    def _render_answer(findings: list[ContractFinding]) -> str:
        if not findings:
            return "No contracts in the corpus match that criterion."
        lines = [f"{len(findings)} contract(s) match:"]
        for f in findings:
            lines.append(f"- {f.doc_id}: {f.reason}")
        return "\n".join(lines)
