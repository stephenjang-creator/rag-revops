"""Tests for analytical (cross-corpus) mode: set-based P/R/F1 and golden builder.

The retriever/generator need live API calls, so their pure logic is tested via
the sandbox harness; here we lock in the deterministic pieces that gate CI: the
precision/recall/F1 computation and the golden-set builder's filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.build_analytical_golden import build as build_analytical
from eval.run_analytical_eval import _prf


def test_prf_perfect():
    assert _prf({"A", "B"}, {"A", "B"}) == (1.0, 1.0, 1.0)


def test_prf_partial():
    p, r, f = _prf({"A", "B", "X"}, {"A", "B", "C"})
    assert abs(p - 2 / 3) < 1e-9
    assert abs(r - 2 / 3) < 1e-9


def test_prf_high_precision_low_recall():
    p, r, f = _prf({"A"}, {"A", "B", "C"})
    assert p == 1.0
    assert abs(r - 1 / 3) < 1e-9


def test_prf_all_false_positives():
    assert _prf({"X", "Y"}, {"A"}) == (0.0, 0.0, 0.0)


def test_prf_empty_prediction():
    p, r, f = _prf(set(), {"A", "B"})
    assert p == 0.0 and r == 0.0


def test_prf_both_empty_is_perfect():
    # A question with no gold matches, and we predicted none — that's correct.
    assert _prf(set(), set()) == (1.0, 1.0, 1.0)


def test_analytical_golden_min_contracts_filter(tmp_path: Path):
    seed = tmp_path / "seed.jsonl"
    out = tmp_path / "ag.jsonl"
    records = [
        {"id": "C1__Governing Law", "contract_file": "C1.txt"},
        {"id": "C2__Governing Law", "contract_file": "C2.txt"},
        {"id": "C3__Governing Law", "contract_file": "C3.txt"},
        {"id": "C1__Non-Compete", "contract_file": "C1.txt"},  # only 1 -> excluded
    ]
    with seed.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    build_analytical(seed, out, min_contracts=3)
    items = [json.loads(l) for l in out.open()]
    categories = {i["category"] for i in items}
    assert "Governing Law" in categories
    assert "Non-Compete" not in categories
    gl = next(i for i in items if i["category"] == "Governing Law")
    assert set(gl["gold_contracts"]) == {"C1.txt", "C2.txt", "C3.txt"}
