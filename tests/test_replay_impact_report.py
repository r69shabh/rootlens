"""Replay caching, business-impact estimation, markdown report."""

import json

import pytest

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig
from data_engine.scenarios import get_scenario
from diagnosis.agent import diagnose
from diagnosis.impact import MANUAL_BASELINE_MINUTES, estimate_impact
from diagnosis.llm_client import ScriptedLLMClient, _chat_with_retry
from diagnosis.replay import (
    RecordingLLMClient,
    ReplayCache,
    ReplayLLMClient,
)
from eval.harness import score_result
from eval.report import to_markdown

_WC = WindowConfig(start=DEFAULT_WINDOW_START)
DIAG = dict(
    current_start=_WC.current_window_start, current_end=_WC.current_window_end,
    baseline_start=_WC.start, baseline_end=_WC.current_window_start,
)


def test_recording_then_replay_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    scripted = ScriptedLLMClient(["hello", "world"])
    rec = RecordingLLMClient(scripted, ReplayCache(cache_path))
    q1 = [{"role": "user", "content": "q1"}]
    q2 = [{"role": "user", "content": "q2"}]
    assert rec.chat("sys", q1) == "hello"
    assert rec.chat("sys", q2) == "world"
    rec.cache.save()
    assert len(ReplayCache(cache_path)) == 2

    replay = ReplayLLMClient(ReplayCache(cache_path))
    assert replay.chat("sys", q1) == "hello"  # deterministic per prompt
    assert replay.chat("sys", q2) == "world"


def test_replay_miss_is_loud(tmp_path):
    cache_path = tmp_path / "empty.json"
    cache_path.write_text("{}")
    replay = ReplayLLMClient(ReplayCache(cache_path))
    with pytest.raises(LookupError, match="cache miss"):
        replay.chat("sys", [{"role": "user", "content": "never seen"}])


def test_chat_with_retry_succeeds_after_transient_error():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert _chat_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_chat_with_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="failed after"):
        _chat_with_retry(always_fail)
    assert calls["n"] == 3


def test_chat_with_retry_does_not_swallow_value_error():
    # ValueError is the agent-loop contract violation signal; must propagate
    def bad():
        raise ValueError("malformed tool call")
    with pytest.raises(ValueError, match="malformed"):
        _chat_with_retry(bad)


def test_replay_caches_full_agent_transcript(tmp_path):
    cache_path = tmp_path / "agent_cache.json"
    responses = [
        json.dumps({"tool": "query_transactions", "args": {"metrics": ["count"]}}),
        json.dumps({"verdict": {"root_cause": "x:y", "confidence": 0.9,
                                "evidence": ["call_001"],
                                "disconfirmation": ["checked: fine"]}}),
    ]
    con = get_scenario("healthy").build_dataset()[0]
    rec = RecordingLLMClient(ScriptedLLMClient(responses), ReplayCache(cache_path))
    r1 = diagnose(con, llm=rec, scenario_id="replay", **DIAG)
    rec.cache.save()
    assert r1.status == "verdict"
    r2 = diagnose(con, llm=ReplayLLMClient(ReplayCache(cache_path)), scenario_id="replay", **DIAG)
    assert r2.to_json() == r1.to_json()


def test_impact_estimate_honest_bounds(tmp_path=None):
    con = get_scenario("bank_outage_icici").build_dataset()[0]
    est = estimate_impact(con, _WC.current_window_start, _WC.current_window_end, 0.5)
    assert est["non_success_txns_in_window"] > 50
    assert est["gmv_at_risk_inr"] > 0
    assert est["manual_baseline_minutes"] == MANUAL_BASELINE_MINUTES
    assert est["hours_saved_vs_manual"] == pytest.approx(
        round((MANUAL_BASELINE_MINUTES - 0.5) / 60, 2), abs=1e-9)
    assert "upper bound" in est["note"]


def test_markdown_report_renders_verdict_and_evidence():
    con = get_scenario("healthy").build_dataset()[0]
    responses = [
        json.dumps({"tool": "query_transactions", "args": {"metrics": ["count"]}}),
        json.dumps({"verdict": {"root_cause": "bank_outage:ICICI", "confidence": 0.8,
                                "evidence": ["call_001"],
                                "disconfirmation": ["compared issuers: isolated to ICICI"]}}),
    ]
    result = diagnose(con, llm=ScriptedLLMClient(responses), scenario_id="md", **DIAG)
    md = to_markdown(result, store=result.store,
                     ground_truth={"expected_labels": ["bank_outage:ICICI"]})
    assert "bank_outage:ICICI" in md
    assert "call_001" in md
    assert "query_transactions" in md
    assert "compared issuers" in md


def test_agent_attaches_impact_to_result():
    con = get_scenario("healthy").build_dataset()[0]
    responses = [
        json.dumps({"tool": "query_transactions", "args": {"metrics": ["count"]}}),
        json.dumps({"verdict": {"root_cause": "x:y", "confidence": 0.9,
                                "evidence": [], "disconfirmation": ["d"]}}),
        json.dumps({"verdict": {"root_cause": "x:y", "confidence": 0.9,
                                "evidence": ["call_001"], "disconfirmation": ["d"],
                                "impact": {"transactions_affected": 5}}}),
    ]
    result = diagnose(con, llm=ScriptedLLMClient(responses), scenario_id="imp", **DIAG)
    assert "estimated" in result.impact
    assert result.impact["claimed_by_model"]["transactions_affected"] == 5
    assert result.time_to_diagnosis_minutes is not None
    score = score_result(result, {"scenario_id": "s", "difficulty_tier": "clean",
                                  "expected_labels": ["x:y"],
                                  "expected_fault_types": ["x"]})
    assert score["correct"] is True
