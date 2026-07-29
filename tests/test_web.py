"""Tests for the web service — keyless endpoints and the contract parser.

The live /api/ask path needs real API keys and a running pipeline, so it isn't
exercised here beyond the no-keys guard. Skips cleanly if the optional web extra
(fastapi) isn't installed, so CI without ".[web]" doesn't fail.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from web.contracts import parse_contract  # noqa: E402
from web.demo_data import demo_runs  # noqa: E402
from web.server import app  # noqa: E402

client = TestClient(app)


# --- contract-id parsing -----------------------------------------------------
def test_parse_contract_extracts_title_and_year():
    m = parse_contract("041__NICELTD_2003-EX-4.5-OUTSOURCING_AGREEMENT")
    assert m.year == 2003
    assert "Outsourcing Agreement" in m.title
    assert m.doc_id.startswith("041__")


def test_parse_contract_without_year_is_graceful():
    m = parse_contract("SomeParty-MASTER_AGREEMENT")
    assert m.year is None
    assert m.title  # non-empty, doesn't crash


# --- /api/demo ---------------------------------------------------------------
def test_demo_endpoint_returns_three_runs_in_event_shape():
    r = client.get("/api/demo")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert {run["id"] for run in runs} == {"draft", "find", "decline"}
    for run in runs:
        assert run["question"]
        types = [e["type"] for e in run["events"]]
        assert types[0] == "stage" and types[-1] == "done"
        assert "route" in types and "rewrite" in types


def test_demo_runs_terminal_event_matches_result_kind():
    by_id = {r["id"]: r for r in demo_runs()}
    assert any(e["type"] == "answer" and e["data"]["kind"] == "draft"
               for e in by_id["draft"]["events"])
    assert any(e["type"] == "answer" and e["data"]["kind"] == "find"
               for e in by_id["find"]["events"])
    assert any(e["type"] == "decline" for e in by_id["decline"]["events"])


# --- /api/health -------------------------------------------------------------
def test_health_reports_index_status():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"status", "corpus_size", "index_ready", "config_version"}


# --- /api/ask guard ----------------------------------------------------------
def test_ask_without_keys_streams_keys_required_error():
    r = client.post("/api/ask", json={"question": "hello", "mode": "auto"})
    assert r.status_code == 200
    assert "event: error" in r.text
    assert "keys_required" in r.text


def test_ask_without_question_is_400():
    r = client.post("/api/ask", json={"question": "   ", "mode": "auto"})
    assert r.status_code == 400
