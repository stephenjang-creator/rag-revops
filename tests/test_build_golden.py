"""Tests for the golden dataset builder: grouping, span dedup, stratification."""

from __future__ import annotations

import json
from pathlib import Path

from eval.build_golden import build
from eval.question_templates import question_for


def _write_seed(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _seed_record(contract: str, category: str, answer: str, start: int) -> dict:
    return {
        "id": f"{contract}__{category}",
        "contract_title": contract,
        "contract_file": f"{contract}.txt",
        "clause_question": f"Highlight parts related to {category}",
        "gold_answer": answer,
        "answer_start": start,
    }


def test_groups_spans_by_contract_and_category(tmp_path: Path):
    seed = tmp_path / "seed.jsonl"
    out = tmp_path / "golden.jsonl"
    _write_seed(
        seed,
        [
            _seed_record("C1", "Parties", "Alpha Corp", 10),
            _seed_record("C1", "Parties", "Beta LLC", 40),   # same clause, 2nd span
            _seed_record("C1", "Governing Law", "New York", 90),
        ],
    )
    build(seed, out, max_per_category=10, limit=0)
    items = [json.loads(line) for line in out.open()]

    parties = next(i for i in items if i["category"] == "Parties")
    assert set(parties["gold_spans"]) == {"Alpha Corp", "Beta LLC"}
    # A natural question replaced the raw extraction prompt.
    assert parties["question"] == question_for("Parties")
    assert "Highlight" not in parties["question"]


def test_dedupes_identical_spans(tmp_path: Path):
    seed = tmp_path / "seed.jsonl"
    out = tmp_path / "golden.jsonl"
    _write_seed(
        seed,
        [
            _seed_record("C1", "Insurance", "Carry $1M coverage", 5),
            _seed_record("C1", "Insurance", "Carry $1M coverage", 5),  # duplicate
        ],
    )
    build(seed, out, max_per_category=10, limit=0)
    item = json.loads(out.open().readline())
    assert item["gold_spans"] == ["Carry $1M coverage"]


def test_stratification_caps_per_category(tmp_path: Path):
    seed = tmp_path / "seed.jsonl"
    out = tmp_path / "golden.jsonl"
    # 5 different contracts all with a "Parties" clause; cap at 2.
    _write_seed(
        seed,
        [_seed_record(f"C{i}", "Parties", f"Party {i}", i) for i in range(5)],
    )
    build(seed, out, max_per_category=2, limit=0)
    items = [json.loads(line) for line in out.open()]
    assert len(items) == 2


def test_limit_truncates_total(tmp_path: Path):
    seed = tmp_path / "seed.jsonl"
    out = tmp_path / "golden.jsonl"
    _write_seed(
        seed,
        [_seed_record(f"C{i}", cat, "span", i)
         for i in range(3)
         for cat in ("Parties", "Insurance", "Governing Law")],
    )
    build(seed, out, max_per_category=10, limit=4)
    items = [json.loads(line) for line in out.open()]
    assert len(items) == 4
