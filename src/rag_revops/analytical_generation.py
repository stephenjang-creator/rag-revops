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
from .observability import (
    RequestTimer,
    count_llm_call,
    flush,
    observe,
    record,
    record_usage,
    trace_context,
)

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

    @observe("analytical_generate")
    def generate(self, question: str, matches: list[ContractMatch]) -> AnalyticalResult:
        with (
            trace_context("analytical_query", tags=["cross-corpus"], question=question,
                          config_version=self.settings.version),
            RequestTimer(mode="analytical", question=question) as timer,
        ):
            result = self._generate_inner(question, matches)
            # Analytical answers are set-membership results; "grounded" here
            # means the query produced >=1 confirmed finding (with a cited
            # reason). No findings on a real query is a decline-like outcome.
            declined = not result.findings
            timer.set_result(
                declined=declined,
                n_citations=len(result.findings),
                n_context_chunks=result.n_candidates,
            )
        flush()
        return result

    def _generate_inner(
        self, question: str, matches: list[ContractMatch]
    ) -> AnalyticalResult:
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

        # Generate equivalent-phrasings guidance ONCE per query, so the judge
        # generalizes to any criterion (payment terms, audit rights, etc.) rather
        # than being hardcoded to one clause type. One cheap Claude call, reused
        # across all judging batches.
        guidance = self._criterion_guidance(question, model)

        batch_size = self.settings.analytical.judge_batch_size

        # Judge in a stable, query-independent order (by doc_id) so batch
        # composition doesn't depend on the rerank ordering. Two equivalent
        # phrasings ("do either parties have a TFC" vs "termination for
        # convenience") rerank the contracts in slightly different orders;
        # batching in that order placed borderline contracts next to different
        # neighbors and yielded different confirm/reject counts (e.g. 7 vs 9).
        # Sorting by doc_id makes the batches identical across phrasings, so the
        # judge sees the same contracts grouped together every time.
        ordered = sorted(matches, key=lambda m: m.doc_id)
        findings: list[ContractFinding] = []
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            findings.extend(self._judge_batch(question, batch, model, guidance))

        # Present the confirmed contracts best-ranked first (rerank score),
        # independent of the doc_id ordering used for judging above.
        findings.sort(key=lambda f: f.score, reverse=True)

        answer = self._render_answer(findings)
        record(
            n_candidates=len(matches),
            n_confirmed=len(findings),
            judge_batches=(len(matches) + batch_size - 1) // batch_size,
        )
        return AnalyticalResult(
            question=question,
            findings=findings,
            answer=answer,
            n_candidates=len(matches),
            model=model,
        )

    def _criterion_guidance(self, question: str, model: str) -> str:
        """Ask the model to enumerate the equivalent ways a contract might express
        the criterion, so the judge recognizes paraphrases for ANY query — not just
        the one clause type we happened to hardcode."""
        count_llm_call()
        resp = self._get_client().messages.create(
            model=model,
            max_tokens=400,
            temperature=0.0,
            system=(
                "You help a contract-analysis system recognize a legal/commercial "
                "concept across different wordings. Given a 'which contracts have X' "
                "question, list the concrete ways a contract might express X — "
                "including synonyms, paraphrases, and typical clause language — so a "
                "reader can spot matches that avoid the exact query words. Be specific "
                "and contract-realistic. Output a short bulleted list, nothing else."
            ),
            messages=[{"role": "user", "content": f"Question: {question}"}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    @observe("membership_judge", as_type="generation")
    def _judge_batch(
        self, question: str, batch: list[ContractMatch], model: str, guidance: str
    ) -> list[ContractFinding]:
        cfg = self.settings.generation
        candidates = _format_batch(batch)
        system = (
            "You are Deal Desk Helper analytical mode, judging which contracts satisfy a "
            "criterion. You receive numbered CANDIDATE contracts with excerpts. Your "
            "job is to CONFIRM every candidate whose excerpts satisfy the criterion — "
            "be thorough, not stingy; a genuine match that you miss is a failure.\n\n"
            "Judge by MEANING, not exact wording. A clause counts even if it never "
            "uses the query's exact words, as long as it expresses the concept. Here "
            "are equivalent ways this criterion may appear in a contract:\n"
            f"{guidance}\n\n"
            "If an excerpt expresses the concept in ANY of these forms (or a clear "
            "equivalent), CONFIRM it. Only drop candidates whose excerpts show no such "
            "language. Do not invent language that isn't there, but DO recognize "
            "equivalent phrasings. Respond with strict JSON only."
        )
        user = (
            f"CRITERION: {question}\n\n"
            f"CANDIDATES:\n{candidates}\n\n"
            'Return JSON exactly: {"matches": [{"n": <candidate number>, '
            '"reason": "<one sentence grounded in that candidate\'s excerpts>"}]}\n'
            "Include only candidates that genuinely satisfy the criterion by meaning. "
            'If none do, return {"matches": []}.'
        )
        count_llm_call()
        resp = self._get_client().messages.create(
            model=model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        findings = self._parse_findings(text, batch)

        usage = getattr(resp, "usage", None)
        record_usage(
            model=model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
        record(
            batch_size=len(batch),
            members=len(findings),
            rejected=len(batch) - len(findings),
        )
        return findings

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
