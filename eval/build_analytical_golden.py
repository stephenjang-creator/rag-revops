"""Build the analytical golden set from CUAD annotations.

For cross-corpus 'which contracts have X' evaluation, the ground truth is a SET of
contracts: for each clause category, exactly which contracts in the corpus have an
annotated (answerable) span for it. CUAD gives us this for free — a contract has a
clause iff it has at least one answerable QA pair for that category.

Output: eval/analytical_golden.jsonl, one record per clause category:
  {
    "question": "Which contracts allow termination for convenience?",
    "category": "Termination For Convenience",
    "gold_contracts": ["012__ACME....txt", "047__BETA....txt", ...]
  }

Only categories with a natural analytical question and at least `min_contracts`
matching contracts are kept (a category matched by a single contract isn't a useful
cross-corpus test).

Usage:
    python -m eval.build_analytical_golden \
        --seed eval/eval_seed.jsonl \
        --out eval/analytical_golden.jsonl \
        --min-contracts 3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Analytical phrasing: "which contracts ..." for each clause category.
ANALYTICAL_QUESTIONS: dict[str, str] = {
    "Termination For Convenience": "Which contracts allow termination for convenience?",
    "Governing Law": "Which contracts specify a governing law?",
    "Cap On Liability": "Which contracts cap liability?",
    "Uncapped Liability": "Which contracts have uncapped or unlimited liability?",
    "Insurance": "Which contracts require a party to carry insurance?",
    "Audit Rights": "Which contracts grant audit rights?",
    "Non-Compete": "Which contracts contain a non-compete restriction?",
    "Exclusivity": "Which contracts grant exclusivity?",
    "Anti-Assignment": "Which contracts restrict assignment?",
    "Change Of Control": "Which contracts have a change-of-control provision?",
    "Minimum Commitment": "Which contracts impose a minimum commitment?",
    "Ip Ownership Assignment": "Which contracts assign intellectual property ownership?",
    "License Grant": "Which contracts grant a license?",
    "Renewal Term": "Which contracts have a renewal term?",
    "Post-Termination Services": "Which contracts require post-termination services?",
    "Revenue/Profit Sharing": "Which contracts include revenue or profit sharing?",
    "Warranty Duration": "Which contracts specify a warranty duration?",
    "No-Solicit Of Employees": "Which contracts restrict soliciting employees?",
    "Non-Transferable License": "Which contracts grant a non-transferable license?",
    "Irrevocable Or Perpetual License": (
        "Which contracts grant an irrevocable or perpetual license?"
    ),
}


def _category_of(record: dict) -> str:
    return record["id"].rsplit("__", 1)[-1]


def build(seed_path: Path, out_path: Path, min_contracts: int) -> None:
    # category -> set of contract files that have an answerable span for it
    category_contracts: dict[str, set[str]] = defaultdict(set)
    with seed_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            category_contracts[_category_of(rec)].add(rec["contract_file"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for category, question in sorted(ANALYTICAL_QUESTIONS.items()):
            contracts = sorted(category_contracts.get(category, set()))
            if len(contracts) < min_contracts:
                continue
            out.write(
                json.dumps(
                    {
                        "question": question,
                        "category": category,
                        "gold_contracts": contracts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"Wrote {n_written} analytical golden items to {out_path}")
    print("(each is a 'which contracts have X' question with its ground-truth set)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, default=Path("eval/eval_seed.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/analytical_golden.jsonl"))
    ap.add_argument("--min-contracts", type=int, default=3)
    args = ap.parse_args()
    build(args.seed, args.out, args.min_contracts)


if __name__ == "__main__":
    main()
