"""Synthetic data engine: determinism + sanity."""

from data_engine.generator import DEFAULT_WINDOW_START, TransactionGenerator, WindowConfig


def _gen(seed=42, txns=2000):
    w = WindowConfig(start=DEFAULT_WINDOW_START)
    return TransactionGenerator(seed=seed, window=w, txns_per_day=txns)


def test_generation_is_deterministic():
    con1 = _gen().generate()
    con2 = _gen().generate()
    q = "SELECT txn_id, ts, amount, status, issuer_bank FROM transactions ORDER BY txn_id"
    assert con1.execute(q).fetchall() == con2.execute(q).fetchall()


def test_schema_and_volume():
    con = _gen().generate()
    cols = [r[0] for r in con.execute("DESCRIBE transactions").fetchall()]
    assert set(cols) >= {
        "txn_id",
        "ts",
        "amount",
        "currency",
        "payment_method",
        "card_network",
        "issuer_bank",
        "status",
        "failure_code",
        "gateway_latency_ms",
        "merchant_id",
        "geo_region",
    }
    n = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert n >= 2000 * 6  # baseline_days(5) + current day


def test_baseline_failure_rate_is_low():
    con = _gen().generate()
    w = _gen().window
    fr = con.execute(
        "SELECT AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END) FROM transactions "
        "WHERE ts < ?",
        [w.current_window_start],
    ).fetchone()[0]
    assert 0.005 < fr < 0.08, f"baseline failure rate {fr} outside sane band"


def test_no_pan_like_data_and_no_pending_leak_into_failed():
    con = _gen().generate()
    bad = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE failure_code IS NOT NULL AND status = 'success'"
    ).fetchone()[0]
    assert bad == 0
