"""Fault injectors: signature strength + scope isolation + ground truth."""

from datetime import UTC

import pytest

from data_engine.faults import (
    BankOutageInjector,
    HighTicketRuleInjector,
    NetworkDegradationInjector,
)
from data_engine.scenarios import get_scenario


def _make_con(scenario_id: str):
    return get_scenario(scenario_id).build_dataset()[0]


def test_bank_outage_hits_only_target_bank():
    con = _make_con("bank_outage_icici")
    from datetime import datetime

    from data_engine.generator import WindowConfig
    wc = WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))
    q = """SELECT issuer_bank, AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END)
           FROM transactions WHERE ts >= ? AND ts < ? GROUP BY 1"""
    rates = dict(con.execute(q, [wc.current_window_start, wc.current_window_end]).fetchall())
    assert rates["ICICI"] > 0.75
    for bank in ["HDFC", "SBI", "AXIS", "KOTAK"]:
        assert rates[bank] < 0.15, f"collateral damage on {bank}: {rates[bank]}"
    codes = dict(con.execute(
        "SELECT failure_code, COUNT(*) FROM transactions WHERE issuer_bank='ICICI' "
        "AND ts >= ? AND ts < ? AND status='failed' GROUP BY 1",
        [wc.current_window_start, wc.current_window_end],
    ).fetchall())
    # outage code dominates; pre-existing baseline codes may remain (realistic)
    assert codes.get("issuer_unavailable", 0) == max(codes.values())
    assert codes.get("issuer_unavailable", 0) > sum(
        v for k, v in codes.items() if k != "issuer_unavailable")


def test_network_degradation_spans_all_issuers():
    con = _make_con("network_degradation_visa")
    from datetime import datetime

    from data_engine.generator import WindowConfig
    wc = WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))
    q = """SELECT issuer_bank, AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END)
           FROM transactions
           WHERE payment_method='card' AND card_network='visa' AND ts >= ? AND ts < ?
           GROUP BY 1"""
    rates = dict(con.execute(q, [wc.current_window_start, wc.current_window_end]).fetchall())
    assert len(rates) >= 4
    assert all(r > 0.25 for r in rates.values()), rates


def test_high_ticket_rule_respects_threshold_and_midwindow_onset():
    con = _make_con("high_ticket_rule_10k")
    from datetime import datetime, timedelta

    from data_engine.generator import WindowConfig
    wc = WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))
    first_hour_end = wc.current_window_start + timedelta(hours=1)
    # before the rule: no excess failures for high amounts
    early_fr = con.execute(
        """SELECT AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END) FROM transactions
           WHERE amount > 10000 AND ts >= ? AND ts < ?""",
        [wc.current_window_start, first_hour_end],
    ).fetchone()[0]
    assert early_fr < 0.25
    late_fr = con.execute(
        """SELECT AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END) FROM transactions
           WHERE amount > 10000 AND ts >= ? AND ts < ?""",
        [first_hour_end, wc.current_window_end],
    ).fetchone()[0]
    assert late_fr > 0.5
    # small amounts untouched
    small_fr = con.execute(
        """SELECT AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END) FROM transactions
           WHERE amount <= 10000 AND ts >= ? AND ts < ?""",
        [first_hour_end, wc.current_window_end],
    ).fetchone()[0]
    assert small_fr < 0.15


def test_compound_produces_two_faults_and_ground_truth():
    gt = get_scenario("compound_outage_plus_rule").ground_truth()
    assert gt["difficulty_tier"] == "compound"
    assert len(gt["expected_labels"]) == 2
    assert "bank_outage:ICICI" in gt["expected_labels"]
    assert any(lb.startswith("rule_trigger:") for lb in gt["expected_labels"])


def test_injector_validates_inputs():
    from datetime import datetime

    from data_engine.generator import WindowConfig
    wc = WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))
    with pytest.raises(ValueError):
        BankOutageInjector(wc, "NOT_A_BANK")
    with pytest.raises(ValueError):
        NetworkDegradationInjector(wc, "visa", failure_rate=0.9)
    with pytest.raises(ValueError):
        HighTicketRuleInjector(wc, 10000, failure_rate=0.2)


def test_ground_truth_records_fault_events_table():
    con, faults = get_scenario("bank_outage_icici").build_dataset()
    rows = con.execute("SELECT fault_type, difficulty_tier FROM fault_events").fetchall()
    assert rows == [("bank_outage", "clean")]
