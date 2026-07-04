"""Extract the CUAD corpus + evaluation seed from CUADv1.json.

Turns the SQuAD-style CUAD file into the two things the RAG pipeline needs:

  1. data/raw/contracts/*.txt   — one plaintext contract per file (the corpus
                                  your chunker + ingest consumes)
  2. eval/eval_seed.jsonl       — answerable QA pairs (question, gold answer text,
                                  answer_start, source contract) for Phase 3

Usage:
    python scripts/extract_cuad.py \
        --input /mnt/user-data/uploads/CUADv1.json \
        --contracts-out data/raw/contracts \
        --eval-out eval/eval_seed.jsonl \
        --max-contracts 0            # 0 = all 510; set e.g. 18 for a demo subset
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(title: str, index: int) -> str:
    """CUAD titles can be long and messy; make a stable, filesystem-safe name."""
    stem = _SAFE.sub("_", title).strip("_")[:120] or f"contract_{index}"
    return f"{index:03d}__{stem}.txt"


def extract(input_path: Path, contracts_out: Path, eval_out: Path, max_contracts: int) -> None:
    with input_path.open(encoding="utf-8") as fh:
        data = json.load(fh)["data"]

    if max_contracts > 0:
        data = data[:max_contracts]

    contracts_out.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)

    n_contracts = 0
    n_pairs = 0

    with eval_out.open("w", encoding="utf-8") as ev:
        for i, contract in enumerate(data):
            title = contract["title"]
            filename = _safe_filename(title, i)

            # A CUAD contract has a single paragraph whose `context` is the full text.
            context_parts = [p["context"] for p in contract["paragraphs"]]
            full_text = "\n\n".join(context_parts).strip()

            (contracts_out / filename).write_text(full_text, encoding="utf-8")
            n_contracts += 1

            # Pull answerable QA pairs for the eval seed.
            for para in contract["paragraphs"]:
                for qa in para["qas"]:
                    if qa["is_impossible"] or not qa["answers"]:
                        continue
                    # De-dup identical answer spans (CUAD sometimes repeats).
                    seen: set[tuple[str, int]] = set()
                    for ans in qa["answers"]:
                        key = (ans["text"], ans["answer_start"])
                        if key in seen:
                            continue
                        seen.add(key)
                        record = {
                            "id": qa["id"],
                            "contract_title": title,
                            "contract_file": filename,
                            "clause_question": qa["question"],
                            "gold_answer": ans["text"],
                            "answer_start": ans["answer_start"],
                        }
                        ev.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_pairs += 1

    print(f"Wrote {n_contracts} contracts to {contracts_out}")
    print(f"Wrote {n_pairs} answerable QA pairs to {eval_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--contracts-out", type=Path, default=Path("data/raw/contracts"))
    ap.add_argument("--eval-out", type=Path, default=Path("eval/eval_seed.jsonl"))
    ap.add_argument("--max-contracts", type=int, default=0)
    args = ap.parse_args()
    extract(args.input, args.contracts_out, args.eval_out, args.max_contracts)


if __name__ == "__main__":
    main()
