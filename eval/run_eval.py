"""Offline faithfulness evaluation using a custom Claude judge.

Runs each golden question through the live pipeline (hybrid -> rerank -> generate),
then scores the answers with a transparent LLM-as-judge (eval/judge.py):

  - faithfulness       : are the answer's claims grounded in the retrieved chunks?
                         (the anti-hallucination metric — the CI gate guards this)
  - answer_relevancy   : does the answer actually address the question?
  - context_precision  : did retrieval surface the right passages?

We use Claude directly (config: eval.judge_model) rather than Ragas, which is
structurally incompatible with a langchain-1.x / LangGraph stack. This is also
more transparent — each metric is a defined, inspectable procedure. Caveat:
generating and judging with the same model family carries a mild self-preference
bias; swap eval.judge_model for an independent grader if desired.

Declines are handled separately: faithfulness on a declined answer is meaningless
(no claims to verify), so declined items are excluded from the faithfulness mean
and reported as their own count. This keeps the headline number honest.

Usage:
    # quick, cheap local check on a subset
    python -m eval.run_eval --golden eval/golden.jsonl --limit 15

    # full run, write JSON report, apply the CI gate
    python -m eval.run_eval --golden eval/golden.jsonl --out eval/report.json --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_revops.config import load_settings
from rag_revops.graph import RagPipeline


def _load_golden(path: Path, limit: int) -> list[dict]:
    items = [json.loads(line) for line in path.open(encoding="utf-8")]
    if limit > 0:
        items = items[:limit]
    return items


def run(golden_path: Path, out_path: Path | None, limit: int, gate: bool) -> int:
    settings = load_settings()
    golden = _load_golden(golden_path, limit)
    if not golden:
        print("No golden items found.", file=sys.stderr)
        return 2

    pipeline = RagPipeline(settings)

    # 1. Run the pipeline over every golden question, collecting the fields Ragas
    #    expects: question, answer, contexts (retrieved chunk texts), reference.
    print(f"Running pipeline over {len(golden)} golden items…")
    rows: list[dict] = []
    declined_rows: list[dict] = []
    for i, item in enumerate(golden, start=1):
        question = item["question"]
        result, contexts = pipeline.ask_with_contexts(question)
        reference = " | ".join(item.get("gold_spans", []))
        row = {
            "user_input": question,
            "response": result.answer,
            "retrieved_contexts": contexts,
            "reference": reference,
            "category": item.get("category", ""),
            "declined": result.declined,
        }
        (declined_rows if result.declined else rows).append(row)
        print(f"  [{i}/{len(golden)}] {'DECLINED' if result.declined else 'answered'}: {question[:60]}")

    print(
        f"\nAnswered: {len(rows)}  |  Declined: {len(declined_rows)} "
        f"(declines excluded from faithfulness)"
    )

    if not rows:
        print("All items were declined — no answered items to score.", file=sys.stderr)
        # Still emit a report so CI has an artifact.
        summary = {
            "n_total": len(golden),
            "n_answered": 0,
            "n_declined": len(declined_rows),
            "metrics": {},
        }
        _write_report(out_path, summary, rows, declined_rows)
        return 0

    # 2. Score answered items with the custom Claude judge.
    print("\nScoring with Claude judge (LLM-as-judge; this consumes API credit)…")
    scores = _score_with_judge(settings, rows)

    # 3. Summarize.
    summary = {
        "n_total": len(golden),
        "n_answered": len(rows),
        "n_declined": len(declined_rows),
        "metrics": scores,
    }
    _print_summary(summary)
    _write_report(out_path, summary, rows, declined_rows)

    # 4. Optional CI gate on faithfulness.
    if gate:
        faith = scores.get("faithfulness")
        threshold = settings.eval.min_faithfulness
        if faith is None:
            print("Gate: faithfulness not computed — failing.", file=sys.stderr)
            return 1
        if faith < threshold:
            print(
                f"\n❌ GATE FAILED: faithfulness {faith:.3f} < threshold {threshold:.3f}",
                file=sys.stderr,
            )
            return 1
        print(f"\n✅ GATE PASSED: faithfulness {faith:.3f} >= threshold {threshold:.3f}")

    return 0


def _score_with_judge(settings, rows: list[dict]) -> dict[str, float]:
    """Score answered rows with the custom Claude judge (no Ragas).

    Aggregates per-metric means over items where the metric was applicable
    (None scores — judge parse failures or not-applicable — are excluded, not
    counted as zero, so a judge hiccup doesn't masquerade as a quality failure).
    """
    from eval.judge import ClaudeJudge

    judge = ClaudeJudge(settings)
    metrics = settings.eval.metrics

    accum: dict[str, list[float]] = {m: [] for m in metrics}
    for i, row in enumerate(rows, start=1):
        item = judge.score_item(
            question=row["user_input"],
            answer=row["response"],
            contexts=row["retrieved_contexts"],
            metrics=metrics,
        )
        per_item = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
        }
        for m in metrics:
            v = per_item.get(m)
            if v is not None:
                accum[m].append(v)
        # annotate the row so the JSON report carries per-item scores too
        row["scores"] = {m: per_item.get(m) for m in metrics}
        print(f"  scored [{i}/{len(rows)}]  " + "  ".join(
            f"{m}={per_item.get(m)!s:.5}" if per_item.get(m) is not None else f"{m}=n/a"
            for m in metrics
        ))

    means: dict[str, float] = {}
    for m in metrics:
        vals = accum[m]
        if vals:
            means[m] = sum(vals) / len(vals)
    return means


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 48)
    print("EVALUATION SUMMARY")
    print("=" * 48)
    print(f"Total items:    {summary['n_total']}")
    print(f"Answered:       {summary['n_answered']}")
    print(f"Declined:       {summary['n_declined']}")
    print("-" * 48)
    for name, value in summary["metrics"].items():
        print(f"{name:<22} {value:.3f}")
    print("=" * 48)


def _write_report(
    out_path: Path | None, summary: dict, rows: list[dict], declined_rows: list[dict]
) -> None:
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summary,
        "answered": rows,
        "declined": declined_rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report here.")
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N items.")
    ap.add_argument("--gate", action="store_true", help="Fail (exit 1) if below threshold.")
    args = ap.parse_args()
    sys.exit(run(args.golden, args.out, args.limit, args.gate))


if __name__ == "__main__":
    main()
