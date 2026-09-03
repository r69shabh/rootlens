"""Golden-hash determinism: a scenario built from the same seed must always
produce byte-identical rows, on any host. Pins platform-dependent behavior
(hash() stability, NumPy reproducibility) so a CI machine producing a
different digest is loud, not silent."""

import hashlib

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


def test_different_seed_yields_different_dataset():
    a, _ = get_scenario("healthy").build_dataset()
    b, _ = get_scenario("healthy").with_seed(99).build_dataset()
    try:
        assert _digest(a) != _digest(b)
    finally:
        a.close()
        b.close()
