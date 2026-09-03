"""Agent loop with a scripted LLM: protocol, round cap, evidence chain."""

import json

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig
from data_engine.scenarios import get_scenario
from diagnosis.agent import diagnose
from diagnosis.llm_client import ScriptedLLMClient, parse_json_response

WC = WindowConfig(start=DEFAULT_WINDOW_START)
ARGS = dict(
    current_start=WC.current_window_start, current_end=WC.current_window_end,
    baseline_start=WC.start, baseline_end=WC.current_window_start,
)


def test_parse_json_response_handles_fences_and_prose():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Sure! here is my answer: {"a": 1} hope that helps') == {"a": 1}


def test_agent_runs_tool_calls_then_verdict():
    responses = [
        json.dumps({"thought": "check failure codes", "tool": "query_transactions",
                    "args": {"filters": {"status": "failed"}, "group_by": ["failure_code"],
                             "metrics": ["count"], "start": str(WC.current_window_start),
                             "end": str(WC.current_window_end)}}),
        json.dumps({"verdict": {"root_cause": "test_cause:x", "confidence": 0.9,
                                "evidence": ["call_001"],
                                "disconfirmation": ["checked other banks: fine"],
                                "impact": {"transactions_affected": 10}}}),
    ]
    con = get_scenario("healthy").build_dataset()[0]
    result = diagnose(con, llm=ScriptedLLMClient(responses), scenario_id="t1", **ARGS)
    assert result.status == "verdict"
    assert result.root_cause == "test_cause:x"
    assert result.evidence_call_ids == ["call_001"]
    assert result.store.get("call_001") is not None
    assert result.transcript[-1]["role"] == "assistant"


def test_agent_respects_round_cap_and_goes_inconclusive():
    responses = [json.dumps({"thought": "still digging", "tool": "query_transactions",
                             "args": {"metrics": ["count"]}})] * 12
    con = get_scenario("healthy").build_dataset()[0]
    result = diagnose(con, llm=ScriptedLLMClient(responses), scenario_id="t2", max_rounds=3, **ARGS)
    assert result.status == "inconclusive"
    assert result.rounds_used == 3


def test_agent_survives_tool_errors_and_rejects_empty_verdicts():
    responses = [
        json.dumps({"tool": "query_transactions", "args": {"filters": {"evil": 1}}}),
        json.dumps({"verdict": {"root_cause": "ok:x", "confidence": 0.9,
                                "evidence": [], "disconfirmation": []}}),
        json.dumps({"verdict": {"root_cause": "ok:x", "confidence": 0.9,
                                "evidence": ["call_001"],
                                "disconfirmation": ["checked others: fine"]}}),
    ]
    con = get_scenario("healthy").build_dataset()[0]
    result = diagnose(con, llm=ScriptedLLMClient(responses), scenario_id="t3", **ARGS)
    assert result.status == "verdict"
    assert any("TOOL_ERROR" in str(t.get("result", "")) for t in result.transcript)
    assert any(t.get("role") == "verdict_rejected" for t in result.transcript)
    assert result.rounds_used == 3


def test_inconclusive_is_explicit():
    responses = [json.dumps({"inconclusive": {"missing": "per-attempt gateway logs"}})]
    con = get_scenario("healthy").build_dataset()[0]
    result = diagnose(con, llm=ScriptedLLMClient(responses), scenario_id="t4", **ARGS)
    assert result.status == "inconclusive"
    assert "gateway logs" in result.missing
