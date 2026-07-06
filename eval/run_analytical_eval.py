"""Analytical evaluation: precision / recall / F1 over contract SETS.

For each 'which contracts have X' question, the system returns a set of contracts;
CUAD tells us the true set. So evaluation is set comparison, not LLM judging:

    precision = |predicted ∩ gold| / |predicted|   (did we avoid false positives?)
    recall    = |predicted ∩ gold| / |gold|        (did we find them all?)
    f1        = harmonic mean

This is deterministic and cheap — no judge calls — which makes it ideal for a CI
gate. The only API cost is running the pipeline to get predictions.

Recall is the metric to watch for cross-corpus retrieval: the wide-fetch design
exists to avoid missing matching contracts. Precision reflects the generator's
confirmation step filtering false positives out of the retriever's wide net.

Usage:
    python -m eval.run_analytical_eval \
        --golden eval/analytical_golden.jsonl \
        --out eval/analytical_report.json \
        --limit 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_revops.analytical import AnalyticalRetriever
from rag_revops.analytical_generation import AnalyticalGenerator
from rag_revops.config import load_settings
from rag_revops.embeddings import CohereEmbedder
from rag_revops.vectorstore import ChromaStore


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def run(golden_path: Path, out_path: Path | None, limit: int, gate: bool) -> int:
    settings = load_settings()
    golden = [json.loads(line) for line in golden_path.open(encoding="utf-8")]
    if limit > 0:
        golden = golden[:limit]
    if not golden:
        print("No analytical golden items found.", file=sys.stderr)
        return 2

    store = ChromaStore(settings.vectorstore)
    embedder = CohereEmbedder(settings.embeddings)
    retriever = AnalyticalRetriever(settings, store, embedder)
    generator = AnalyticalGenerator(settings)

    print(f"Running analytical eval over {len(golden)} questions…")
    rows: list[dict] = []
    p_sum = r_sum = f_sum = 0.0
    for i, item in enumerate(golden, start=1):
        question = item["question"]
        gold = set(item["gold_contracts"])

        matches = retriever.retrieve(question)
        result = generator.generate(question, matches)
        predicted = {f.doc_id for f in result.findings}
        # gold_contracts are stored as filenames; doc_id is the filename stem.
        gold_ids = {Path(g).stem for g in gold}

        precision, recall, f1 = _prf(predicted, gold_ids)
        p_sum += precision
        r_sum += recall
        f_sum += f1

        rows.append(
            {
                "question": question,
                "category": item.get("category", ""),
                "n_gold": len(gold_ids),
                "n_predicted": len(predicted),
                "n_correct": len(predicted & gold_ids),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        print(
            f"  [{i}/{len(golden)}] {item.get('category','')[:28]:<28} "
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
            f"(pred {len(predicted)} / gold {len(gold_ids)})"
        )

    n = len(rows)
    summary = {
        "n_questions": n,
        "mean_precision": p_sum / n,
        "mean_recall": r_sum / n,
        "mean_f1": f_sum / n,
    }
    _print_summary(summary)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote report to {out_path}")

    if gate:
        threshold = settings.analytical_eval.min_f1
        if summary["mean_f1"] < threshold:
            print(
                f"\n❌ GATE FAILED: mean F1 {summary['mean_f1']:.3f} < {threshold:.3f}",
                file=sys.stderr,
            )
            return 1
        print(f"\n✅ GATE PASSED: mean F1 {summary['mean_f1']:.3f} >= {threshold:.3f}")

    return 0


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 48)
    print("ANALYTICAL EVALUATION SUMMARY")
    print("=" * 48)
    print(f"Questions:       {summary['n_questions']}")
    print(f"Mean precision:  {summary['mean_precision']:.3f}")
    print(f"Mean recall:     {summary['mean_recall']:.3f}")
    print(f"Mean F1:         {summary['mean_f1']:.3f}")
    print("=" * 48)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=Path("eval/analytical_golden.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    sys.exit(run(args.golden, args.out, args.limit, args.gate))


if __name__ == "__main__":
    main()
