"""Week-3: rule-based baseline agent, leaderboard, eval runner."""

import json
from pathlib import Path

import pytest

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig
from data_engine.scenarios import get_scenario
from diagnosis.baseline_agent import rule_based_diagnose
from eval.harness import score_result
from eval.leaderboard import Leaderboard, RunRecord

WC = WindowConfig(start=DEFAULT_WINDOW_START)
BOUNDS = (WC.current_window_start, WC.current_window_end, WC.start, WC.current_window_start)

CLEAN_FAULT_SCENARIOS = [
    "bank_outage_icici", "bank_outage_kotak", "network_degradation_visa",
    "network_degradation_rupay", "high_ticket_rule_10k", "retry_storm_gateway",
    "checkout_funnel_break", "settlement_delay_mch007",
]


def _gt(sid):
    sc = get_scenario(sid)
    return {"scenario_id": sid, "difficulty_tier": sc.tier,
            "expected_labels": [], "expected_fault_types": []}  # filled below


def _gt_for(sid):
    sc = get_scenario(sid)
    con, faults = sc.build_dataset()
    con.close()
    return {"scenario_id": sid, "difficulty_tier": sc.tier,
            "expected_labels": [f.label for f in faults],
            "expected_fault_types": sorted({f.fault_type for f in faults})}


@pytest.mark.parametrize("sid", CLEAN_FAULT_SCENARIOS + ["compound_outage_plus_rule",
                                                         "noisy_bank_outage_hdfc",
                                                         "noisy_network_amex",
                                                         "red_herring_campaign_vs_outage"])
def test_rule_baseline_diagnoses_all_faults(sid):
    con, _ = get_scenario(sid).build_dataset()
    result = rule_based_diagnose(con, *BOUNDS, scenario_id=sid)
    score = score_result(result, _gt_for(sid))
    assert score["correct"], f"{sid}: {result.root_cause} vs {_gt_for(sid)['expected_labels']}"
    assert result.status == "verdict"
    assert result.evidence_call_ids, "rule agent must cite evidence"
    for cid in result.evidence_call_ids:
        assert result.store.get(cid) is not None
    con.close()


@pytest.mark.parametrize("sid", ["healthy", "benign_volume_spike"])
def test_rule_baseline_stays_quiet_on_controls(sid):
    con, _ = get_scenario(sid).build_dataset()
    result = rule_based_diagnose(con, *BOUNDS, scenario_id=sid)
    assert result.status != "verdict", f"false positive on {sid}: {result.root_cause}"
    score = score_result(result, _gt_for(sid))
    assert score["correct"]
    con.close()


def test_rule_baseline_compound_label_covers_both_families():
    con, _ = get_scenario("compound_outage_plus_rule").build_dataset()
    result = rule_based_diagnose(con, *BOUNDS, scenario_id="c")
    assert "bank_outage" in result.root_cause and "rule_trigger" in result.root_cause


def test_leaderboard_ranking_and_costs():
    board = Leaderboard()
    for agent, correct, lat, toks in [
        ("openai:gpt-4o-mini", True, 2.0, 100_000),
        ("openai:gpt-4o-mini", False, 3.0, 100_000),
        ("rule-baseline", True, 0.0, 0),
        ("rule-baseline", True, 0.0, 0),
    ]:
        board.add(RunRecord(
            scenario_id="s", tier="clean", agent=agent, status="verdict",
            correct=correct, partial=False, inconclusive=False,
            false_positive=False, latency_minutes=lat, llm_calls=1,
            input_tokens=toks, output_tokens=0,
        ))
    standings = board.standings()
    # equal accuracy (100%) -> cheaper agent ranks first
    assert standings[0].agent == "rule-baseline"
    assert standings[1].agent == "openai:gpt-4o-mini"
    assert standings[1].accuracy == 0.5
    # gpt-4o-mini input price: 100k tokens = $0.015 per run
    assert standings[1].est_cost_usd == pytest.approx(0.03, abs=1e-9)
    md = board.to_markdown()
    assert "rule-baseline" in md and "Per-tier accuracy" in md


def test_leaderboard_tiers_never_blend():
    board = Leaderboard()
    for tier, correct in [("clean", True), ("clean", False), ("noisy", True)]:
        board.add(RunRecord(
            scenario_id="s", tier=tier, agent="llm-a", status="verdict",
            correct=correct, partial=False, inconclusive=False,
            false_positive=False, latency_minutes=1.0, llm_calls=1,
        ))
    st = board.standings()[0]
    assert st.per_tier["clean"] == {"total": 2, "correct": 1}
    assert st.per_tier["noisy"] == {"total": 1, "correct": 1}
    assert st.accuracy == pytest.approx(2 / 3)


def test_run_eval_rule_end_to_end(tmp_path, monkeypatch):
    import scripts.run_eval as re_eval
    ids = ["bank_outage_icici", "healthy", "compound_outage_plus_rule"]
    out = tmp_path / "res"
    monkeypatch.chdir(tmp_path)
    re_eval.run("rule", ids, out_dir=str(out))
    files = list(Path(out).glob("*"))
    # timestamped filenames; the prefix is YYYYMMDDTHHMMSSZ_agent
    results = [f for f in files if f.name.endswith("_rule_results.json")]
    leaderboard = [f for f in files if f.name.endswith("_rule_leaderboard.md")]
    assert results, f"no results file in {files}"
    assert leaderboard, f"no leaderboard file in {files}"
    data = json.loads(results[0].read_text())
    by_sid = {d["scenario_id"]: d for d in data["details"]}
    assert by_sid["bank_outage_icici"]["score"]["correct"] is True
    assert by_sid["healthy"]["score"]["correct"] is True  # control: no false alarm
    assert by_sid["compound_outage_plus_rule"]["score"]["correct"] is True
