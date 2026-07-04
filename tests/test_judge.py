"""Tests for the custom Claude judge: JSON parsing and score computation.

The live judge calls need the network + a key, so we stub the model call
(`_ask_json`) and test the scoring math, None-handling, and clamping — the parts
that must be correct for the eval to be trustworthy.
"""

from __future__ import annotations

import pytest

from eval.judge import ClaudeJudge, _parse_json


def _judge():
    from rag_revops.config import load_settings

    return ClaudeJudge(load_settings())


# --- JSON extraction ------------------------------------------------------

def test_parse_json_from_plain_object():
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_from_noisy_text():
    assert _parse_json('Analysis: {"relevancy": 0.9} done')["relevancy"] == 0.9


def test_parse_json_from_markdown_fence():
    assert _parse_json('```json\n{"n_claims": 2}\n```')["n_claims"] == 2


def test_parse_json_returns_none_on_garbage():
    assert _parse_json("no json at all") is None


# --- scoring math (stubbing the model call) -------------------------------

@pytest.fixture
def judge():
    return _judge()


def test_faithfulness_ratio(judge):
    judge._ask_json = lambda s, u: {"n_claims": 3, "n_supported": 2}
    assert abs(judge.faithfulness("ans", ["ctx"]) - 2 / 3) < 1e-9


def test_faithfulness_no_claims_returns_none(judge):
    judge._ask_json = lambda s, u: {"n_claims": 0, "n_supported": 0}
    assert judge.faithfulness("ans", ["ctx"]) is None


def test_faithfulness_parse_failure_returns_none(judge):
    judge._ask_json = lambda s, u: None
    assert judge.faithfulness("ans", ["ctx"]) is None


def test_relevancy_clamps_above_one(judge):
    judge._ask_json = lambda s, u: {"relevancy": 1.5}
    assert judge.answer_relevancy("q", "a") == 1.0


def test_context_precision_ratio(judge):
    judge._ask_json = lambda s, u: {"n_relevant": 1, "n_total": 4}
    assert judge.context_precision("q", ["a", "b", "c", "d"]) == 0.25


def test_empty_inputs_return_none(judge):
    assert judge.faithfulness("", ["ctx"]) is None
    assert judge.context_precision("q", []) is None
