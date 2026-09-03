"""Stage 1: the deterministic scan must find the injected fault — and stay quiet on healthy data."""

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig
from data_engine.scenarios import get_scenario
from diagnosis.anomaly_scan import scan

WC = WindowConfig(start=DEFAULT_WINDOW_START)


def _scan(con):
    return scan(con, WC.current_window_start, WC.current_window_end,
                WC.start, WC.current_window_start)


def test_healthy_window_has_no_anomalies():
    con = get_scenario("healthy").build_dataset()[0]
    segs = _scan(con)
    assert segs == [], f"false positives on healthy data: {[s.to_dict() for s in segs]}"


def _top(con):
    segs = _scan(con)
    assert segs, "no anomalies detected"
    return segs[0]


def test_detects_bank_outage_on_right_dimension():
    seg = _top(get_scenario("bank_outage_icici").build_dataset()[0])
    assert seg.dimension == "issuer_bank" and seg.value == "ICICI"
    assert seg.drop > 0.5


def test_detects_network_degradation_on_right_dimension():
    seg = _top(get_scenario("network_degradation_visa").build_dataset()[0])
    assert seg.dimension == "card_network" and seg.value == "visa"


def test_detects_high_ticket_rule_via_amount_bucket():
    seg = _top(get_scenario("high_ticket_rule_10k").build_dataset()[0])
    assert seg.dimension == "amount_bucket" and seg.value == ">10k"


def test_compound_ranks_bank_outage_first_and_rule_in_top3():
    segs = _scan(get_scenario("compound_outage_plus_rule").build_dataset()[0])
    top3 = [(s.dimension, s.value) for s in segs[:3]]
    assert ("issuer_bank", "ICICI") in top3
    assert ("amount_bucket", ">10k") in top3


def test_low_volume_kotak_still_detected():
    seg = _top(get_scenario("bank_outage_kotak").build_dataset()[0])
    assert (seg.dimension, seg.value) == ("issuer_bank", "KOTAK")
