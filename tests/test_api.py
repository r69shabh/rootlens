"""API smoke tests: import the FastAPI app, drive it with httpx ASGI transport."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_scenarios_list_returns_14(client):
    r = client.get("/scenarios")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 14
    assert {s["id"] for s in data} >= {"bank_outage_icici", "healthy", "compound_outage_plus_rule"}


def test_diagnose_unknown_scenario_is_404(client):
    r = client.post("/diagnose", json={"scenario_id": "not_real", "agent": "rule"})
    assert r.status_code == 404


def test_diagnose_rule_returns_verdict_with_evidence(client):
    r = client.post("/diagnose", json={"scenario_id": "bank_outage_icici", "agent": "rule"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "verdict"
    assert body["root_cause"] == "bank_outage:ICICI"
    assert body["confidence"] is not None
    assert body["evidence_call_ids"], "rule agent must cite at least one evidence call"
    assert body["evidence"], "full audit trail must be present"
    first = body["evidence"][0]
    assert first["call_id"].startswith("call_")
    assert "tool" in first and "args" in first and "result" in first


def test_diagnose_healthy_returns_inconclusive(client):
    r = client.post("/diagnose", json={"scenario_id": "healthy", "agent": "rule"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "inconclusive"
    assert body["root_cause"] is None
