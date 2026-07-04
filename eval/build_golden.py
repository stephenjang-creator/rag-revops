"""Build a curated golden evaluation set from the raw CUAD eval seed.

Input:  eval/eval_seed.jsonl  (from scripts/extract_cuad.py) — one record per
        (contract, clause category, answer span).
Output: eval/golden.jsonl    — one record per (contract, clause category), with:
          - a NATURAL question (see question_templates.py)
          - ALL verified gold answer spans for that category collected together
          - the source contract file

Grouping by (contract, category) matters: CUAD often has several valid spans for
one clause, and an answer that surfaces any of them should count as faithful. The
eval script uses these spans as ground-truth references.

The set is capped and stratified across clause categories so the golden set isn't
dominated by the most common category ("Parties"). This gives broader coverage of
contract concepts, which is what makes a faithfulness score meaningful.

Usage:
    python -m eval.build_golden \
        --seed eval/eval_seed.jsonl \
        --out eval/golden.jsonl \
        --max-per-category 6 \
        --limit 120
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.question_templates import question_for


def _category_of(record: dict) -> str:
    # The extractor encodes category as the segment after the final "__" in id.
    return record["id"].rsplit("__", 1)[-1]


def build(seed_path: Path, out_path: Path, max_per_category: int, limit: int) -> None:
    # Group raw pairs by (contract_file, category), collecting distinct gold spans.
    grouped: dict[tuple[str, str], dict] = {}
    with seed_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            category = _category_of(rec)
            key = (rec["contract_file"], category)
            entry = grouped.setdefault(
                key,
                {
                    "contract_file": rec["contract_file"],
                    "contract_title": rec["contract_title"],
                    "category": category,
                    "gold_spans": [],
                },
            )
            span = rec["gold_answer"].strip()
            if span and span not in entry["gold_spans"]:
                entry["gold_spans"].append(span)

    # Stratify: cap how many items we take per category so the set is balanced.
    by_category: dict[str, list[dict]] = defaultdict(list)
    for entry in grouped.values():
        by_category[entry["category"]].append(entry)

    golden: list[dict] = []
    for category, entries in sorted(by_category.items()):
        for entry in entries[:max_per_category]:
            golden.append(
                {
                    "question": question_for(category),
                    "category": category,
                    "contract_file": entry["contract_file"],
                    "contract_title": entry["contract_title"],
                    "gold_spans": entry["gold_spans"],
                }
            )

    # Deterministic order, then apply the overall limit.
    golden.sort(key=lambda g: (g["category"], g["contract_file"]))
    if limit > 0:
        golden = golden[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for g in golden:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")

    n_categories = len({g["category"] for g in golden})
    print(f"Wrote {len(golden)} golden items across {n_categories} categories to {out_path}")
    print("Category distribution:")
    dist: dict[str, int] = defaultdict(int)
    for g in golden:
        dist[g["category"]] += 1
    for cat, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {n:2d}  {cat}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, default=Path("eval/eval_seed.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--max-per-category", type=int, default=6)
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()
    build(args.seed, args.out, args.max_per_category, args.limit)


if __name__ == "__main__":
    main()
