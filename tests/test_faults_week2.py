"""Week-2 faults: retry storm, checkout funnel, settlement delay, red herrings, noisy."""

from datetime import UTC, datetime

from data_engine.generator import WindowConfig
from data_engine.scenarios import get_scenario
from diagnosis.anomaly_scan import scan

WC = WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))
S, E = WC.current_window_start, WC.current_window_end
MID = S + (E - S) / 2


def _scan(con):
    return scan(con, S, E, WC.start, S)


def _window_metrics(con, extra_where="", params=None):
    rows = con.execute(
        f"""SELECT COUNT(*),
                   AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END),
                   AVG(gateway_latency_ms)
            FROM transactions WHERE ts >= ? AND ts < ? {extra_where}""",
        [S, E, *(params or [])],
    ).fetchone()
    return {"count": rows[0], "failure_rate": rows[1], "avg_latency": rows[2]}


def test_retry_storm_signature():
    con = get_scenario("retry_storm_gateway").build_dataset()[0]
    healthy = get_scenario("healthy").build_dataset()[0]
    m, h = _window_metrics(con), _window_metrics(healthy)
    # volume spike from duplicate retry attempts
    assert m["count"] > h["count"] * 1.15
    # elevated timeouts + latency
    assert m["failure_rate"] > 0.2
    assert m["avg_latency"] > h["avg_latency"] + 400
    timeouts = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE ts >= ? AND ts < ? "
        "AND failure_code = 'gateway_timeout'", [MID, E],
    ).fetchone()[0]
    assert timeouts > 50
    # scan detects a diffuse degradation
    segs = _scan(con)
    assert segs and segs[0].drop > 0.2


def test_checkout_funnel_break_signature():
    con = get_scenario("checkout_funnel_break").build_dataset()[0]
    # all payment methods elevated AFTER onset, untouched before
    early = dict(con.execute(
        """SELECT payment_method,
                  AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END)
           FROM transactions WHERE ts >= ? AND ts < ? GROUP BY 1""", [S, MID]).fetchall())
    late = dict(con.execute(
        """SELECT payment_method,
                  AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END)
           FROM transactions WHERE ts >= ? AND ts < ? GROUP BY 1""", [MID, E]).fetchall())
    for method in early:
        assert late[method] > 0.35, (method, late[method])
        assert early[method] < 0.15, (method, early[method])
    codes = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE ts >= ? AND ts < ? "
        "AND failure_code = 'checkout_error'", [MID, E],
    ).fetchone()[0]
    assert codes > 100


def test_settlement_delay_signature():
    con = get_scenario("settlement_delay_mch007").build_dataset()[0]
    pend = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE merchant_id='mch_007' "
        "AND ts >= ? AND ts < ? AND status='pending'", [MID, E],
    ).fetchone()[0]
    assert pend > 5
    # pending rows carry no failure code — nothing actually failed
    bad = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE merchant_id='mch_007' "
        "AND ts >= ? AND ts < ? AND status='pending' AND failure_code IS NOT NULL",
        [MID, E],
    ).fetchone()[0]
    assert bad == 0
    # other merchants unaffected
    other_pending = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE merchant_id != 'mch_007' "
        "AND ts >= ? AND ts < ? AND status='pending'", [S, E],
    ).fetchone()[0]
    assert other_pending == 0
    # scan surfaces the merchant slice
    seg = _scan(con)[0]
    assert (seg.dimension, seg.value) == ("merchant_id", "mch_007")


def test_noisy_bank_outage_partial_signal():
    con = get_scenario("noisy_bank_outage_hdfc").build_dataset()[0]
    fr = con.execute(
        """SELECT AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END)
           FROM transactions WHERE issuer_bank='HDFC' AND ts >= ? AND ts < ?""",
        [S, E],
    ).fetchone()[0]
    assert 0.45 < fr < 0.75
    codes = {r[0] for r in con.execute(
        """SELECT DISTINCT failure_code FROM transactions
           WHERE issuer_bank='HDFC' AND ts >= ? AND ts < ? AND status='failed'""",
        [S, E]).fetchall()}
    assert "issuer_declined" in codes and "issuer_unavailable" in codes
    seg = _scan(con)[0]
    assert (seg.dimension, seg.value) == ("issuer_bank", "HDFC")


def test_noisy_network_amex_low_volume_detected():
    con = get_scenario("noisy_network_amex").build_dataset()[0]
    seg = _scan(con)[0]
    assert (seg.dimension, seg.value) == ("card_network", "amex")


def test_red_herring_campaign_does_not_mask_real_fault():
    con = get_scenario("red_herring_campaign_vs_outage").build_dataset()[0]
    segs = _scan(con)
    # the REAL fault outranks the benign volume spike
    assert (segs[0].dimension, segs[0].value) == ("issuer_bank", "HDFC")
    assert all(s.value != "north" or s.dimension != "geo_region" for s in segs[:1])
    gt = get_scenario("red_herring_campaign_vs_outage").ground_truth()
    # the benign spike is NOT part of expected labels (it is not a fault)
    assert all("benign" not in lbl for lbl in gt["expected_labels"])


def test_benign_spike_alone_is_not_flagged():
    con = get_scenario("benign_volume_spike").build_dataset()[0]
    assert _scan(con) == [], "false positive on benign volume spike"
    # volume did actually increase (the spike is real, just harmless)
    n_north = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE geo_region='north' AND ts >= ? AND ts < ?",
        [S, E],
    ).fetchone()[0]
    assert n_north > 100
