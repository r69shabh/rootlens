"""Golden-hash determinism: a scenario built from the same seed must always
produce byte-identical rows, on any host. Pins platform-dependent behavior
(hash() stability, NumPy reproducibility) so a CI machine producing a
different digest is loud, not silent."""

import hashlib
from datetime import timedelta

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig
from data_engine.scenarios import get_scenario


def _digest(con) -> str:
    rows = con.execute(
        "SELECT txn_id, ts, amount, status, issuer_bank, payment_method, "
        "card_network, failure_code FROM transactions ORDER BY txn_id"
    ).fetchall()
    return hashlib.sha256(str(rows).encode()).hexdigest()


def test_healthy_dataset_hash_is_pinned():
    con, _ = get_scenario("healthy").build_dataset()
    try:
        assert _digest(con) == "8ac29f6c7cfc2a1db4fbb8012e4670e4ed0781152acc7598c4296ffc4fea1cc7"
    finally:
        con.close()


def test_same_seed_yields_identical_dataset():
    a, _ = get_scenario("healthy").build_dataset()
    b, _ = get_scenario("healthy").build_dataset()
    try:
        assert _digest(a) == _digest(b)
    finally:
        a.close()
        b.close()


def test_query_window_uses_utc_clock_time():
    """Aware UTC bounds must select the stored UTC clock times.

    DuckDB shifts aware parameters into the session TimeZone. A host on
    UTC+5:30 used to read 10:00Z as 15:30 and pass tests CI then failed.
    """
    con, _ = get_scenario("healthy").build_dataset()
    try:
        tz = con.execute("SELECT current_setting('TimeZone')").fetchone()[0]
        assert tz == "UTC"
        wc = WindowConfig(start=DEFAULT_WINDOW_START)
        start = wc.current_window_start.replace(tzinfo=None)
        end = wc.current_window_end.replace(tzinfo=None)
        mn, mx, n = con.execute(
            "SELECT MIN(ts), MAX(ts), COUNT(*) FROM transactions WHERE ts >= ? AND ts < ?",
            [wc.current_window_start, wc.current_window_end],
        ).fetchone()
        assert n > 0
        assert start <= mn < end
        assert start <= mx < end
        # The shifted window begins at 15:30, outside this first hour.
        assert mn < start + timedelta(hours=1)
        assert mx >= end - timedelta(hours=1)
    finally:
        con.close()


def test_different_seed_yields_different_dataset():
    a, _ = get_scenario("healthy").build_dataset()
    b, _ = get_scenario("healthy").with_seed(99).build_dataset()
    try:
        assert _digest(a) != _digest(b)
    finally:
        a.close()
        b.close()
