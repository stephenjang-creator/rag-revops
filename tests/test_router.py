"""Tests for the skill router — classification (stubbed LLM) and contract resolution.

The routing call is a forced tool-use request; we stub the Anthropic client to
return a chosen tool_use block, so these lock in the parsing/defaulting logic and
the deterministic contract resolver with no network. Metrics sink disabled.
"""

from __future__ import annotations

import pytest

from rag_revops.config import load_settings
from rag_revops.router import (
    SKILL_ASK_ONE,
    SKILL_DRAFT,
    SKILL_FIND_MANY,
    QueryRouter,
    resolve_contract,
)


class _ToolUse:
    type = "tool_use"
    name = "route"

    def __init__(self, payload: dict):
        self.input = payload


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Resp:
    def __init__(self, payload: dict):
        self.content = [_ToolUse(payload)]
        self.usage = _Usage()


class _Messages:
    def __init__(self, payload: dict):
        self._payload = payload

    def create(self, **kwargs):
        return _Resp(self._payload)


class _Client:
    def __init__(self, payload: dict):
        self.messages = _Messages(payload)


@pytest.fixture(autouse=True)
def _no_metrics_sink(monkeypatch):
    monkeypatch.setenv("RAG_METRICS_PATH", "")


def _router(payload: dict) -> QueryRouter:
    r = QueryRouter(load_settings())
    r._client = _Client(payload)  # inject stub; _get_client returns it as-is
    return r


def test_routes_single_contract_with_hint():
    d = _router({
        "skill": "ask_one_contract",
        "contract_hint": "Pizza Fusion",
        "needs_clarification": False,
        "reason": "Names a specific contract.",
    }).route("What does the Pizza Fusion agreement say about termination?")
    assert d.skill == SKILL_ASK_ONE
    assert d.contract_hint == "Pizza Fusion"
    assert d.needs_clarification is False


def test_routes_find_contracts():
    d = _router({
        "skill": "find_contracts",
        "contract_hint": "",
        "needs_clarification": False,
        "reason": "Asks which contracts have a clause.",
    }).route("Which contracts allow termination for convenience?")
    assert d.skill == SKILL_FIND_MANY
    assert d.contract_hint is None  # empty string normalizes to None


def test_routes_draft_clause():
    d = _router({
        "skill": "draft_clause",
        "needs_clarification": False,
        "reason": "Wants new reusable language.",
    }).route("Draft a mutual NDA clause")
    assert d.skill == SKILL_DRAFT


def test_unknown_skill_falls_back_to_clarify():
    d = _router({
        "skill": "something_else",
        "needs_clarification": False,
        "reason": "n/a",
    }).route("???")
    assert d.needs_clarification is True


def test_missing_tool_use_block_is_clarify():
    r = QueryRouter(load_settings())

    class _EmptyResp:
        content: list = []
        usage = _Usage()

    class _EmptyClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                return _EmptyResp()

    r._client = _EmptyClient()
    d = r.route("hello")
    assert d.needs_clarification is True


# --- contract resolution -----------------------------------------------------

CONTRACTS = [
    "030__PfHospitalityGroupInc_20150923_10-12G_EX-10.1_Franchise_Agreement1",
    "000__LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR_AGREEMENT",
]


def test_resolve_contract_substring_match():
    assert resolve_contract("Pizza Fusion", CONTRACTS) is None  # not in id text
    assert resolve_contract("PfHospitality", CONTRACTS) == CONTRACTS[0]
    assert resolve_contract("lime energy", CONTRACTS) == CONTRACTS[1]


def test_resolve_contract_none_when_no_match():
    assert resolve_contract("Acme Corp", CONTRACTS) is None
    assert resolve_contract(None, CONTRACTS) is None
    assert resolve_contract("anything", []) is None
