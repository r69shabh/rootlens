"""Deterministic, seeded synthetic transaction data (no faults applied)."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import duckdb
import numpy as np

ISSUER_BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK"]
CARD_NETWORKS = ["visa", "mastercard", "rupay", "amex"]
PAYMENT_METHODS = ["card", "upi", "netbanking"]
GEO_REGIONS = ["north", "south", "west", "east"]
FAILURE_CODES_BY_METHOD = {
    "card": ["insufficient_funds", "issuer_declined", "network_timeout"],
    "upi": ["vpa_invalid", "beneficiary_bank_timeout"],
    "netbanking": ["bank_timeout", "user_abandoned"],
}

BASE_FAILURE_RATE = 0.03

# Single source of truth for the canonical window start. Every scenario, script,
# UI entrypoint, and test must use this instead of a hardcoded datetime, so a
# window change cannot silently desync ground truth from diagnosis.
DEFAULT_WINDOW_START = datetime(2026, 8, 24, tzinfo=UTC)

AMOUNT_BUCKETS = [
    ("<500", 0.0, 500.0),
    ("500-2k", 500.0, 2000.0),
    ("2k-10k", 2000.0, 10000.0),
    (">10k", 10000.0, float("inf")),
]


def amount_bucket(amount: float) -> str:
    for label, low, high in AMOUNT_BUCKETS:
        if low <= amount < high:
            return label
    return ">10k"


@dataclass(frozen=True)
class WindowConfig:
    """Time layout: `baseline_days` of history plus a 2h 'current' window."""

    start: datetime
    baseline_days: int = 5
    current_window_start: datetime = field(init=False)
    current_window_end: datetime = field(init=False)

    def __post_init__(self) -> None:
        cur_start = self.start + timedelta(days=self.baseline_days, hours=10)
        object.__setattr__(self, "current_window_start", cur_start)
        object.__setattr__(self, "current_window_end", cur_start + timedelta(hours=2))

    @property
    def end(self) -> datetime:
        return self.current_window_end

    def bounds(self) -> WindowBounds:
        """Named window bounds: baseline-first order, no positional tuples."""
        return WindowBounds(
            baseline_start=self.start,
            baseline_end=self.current_window_start,
            current_start=self.current_window_start,
            current_end=self.current_window_end,
        )


@dataclass(frozen=True)
class WindowBounds:
    """The four timestamps every scan/diagnosis call needs, by name."""

    baseline_start: datetime
    baseline_end: datetime
    current_start: datetime
    current_end: datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    txn_id VARCHAR,
    ts TIMESTAMP,
    amount DOUBLE,
    currency VARCHAR,
    payment_method VARCHAR,
    card_network VARCHAR,
    issuer_bank VARCHAR,
    status VARCHAR,
    failure_code VARCHAR,
    gateway_latency_ms INTEGER,
    merchant_id VARCHAR,
    geo_region VARCHAR
);
CREATE TABLE IF NOT EXISTS fault_events (
    fault_id VARCHAR,
    fault_type VARCHAR,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    affected_scope JSON,
    difficulty_tier VARCHAR
);
"""


COLUMNS = [
    "txn_id", "ts", "amount", "currency", "payment_method", "card_network",
    "issuer_bank", "status", "failure_code", "gateway_latency_ms",
    "merchant_id", "geo_region",
]


class TransactionGenerator:
    """Seeded generator producing the baseline `transactions` table in DuckDB.

    Deterministic: same (seed, txns_per_day) always yields identical rows.
    """

    def __init__(self, seed: int, window: WindowConfig | None = None, txns_per_day: int = 4000):
        self.seed = seed
        self.window = window or WindowConfig(start=DEFAULT_WINDOW_START)
        self.txns_per_day = txns_per_day
        self.rng = np.random.default_rng(seed)

    def _daily_volume(self, day_index: int) -> int:
        date = (self.window.start + timedelta(days=day_index)).date()
        weekend_factor = 1.25 if date.weekday() >= 5 else 1.0
        return int(self.txns_per_day * weekend_factor)

    def _generate_rows(self, day_index: int) -> list[tuple]:
        """Vectorized row generation for one day; returns tuples in COLUMNS order."""
        rng = self.rng
        day_start = self.window.start + timedelta(days=day_index)
        n = self._daily_volume(day_index)

        hour_w = np.array(
            [2, 2, 1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 7, 7, 8, 9, 9, 8, 7, 6, 4, 3, 2],
            dtype=float,
        )
        hour_w /= hour_w.sum()
        hours = rng.choice(24, size=n, p=hour_w)
        seconds = rng.integers(0, 3600, size=n)
        base = np.datetime64(day_start.replace(tzinfo=None))
        ts = base + hours.astype("timedelta64[h]") + seconds.astype("timedelta64[s]")
        ts_py = ts.astype("datetime64[us]").astype(object)

        methods = rng.choice(PAYMENT_METHODS, size=n, p=[0.70, 0.25, 0.05])
        networks = rng.choice(CARD_NETWORKS, size=n, p=[0.45, 0.35, 0.15, 0.05])
        card_network = np.where(methods == "card", networks, None)

        banks = rng.choice(ISSUER_BANKS, size=n, p=[0.30, 0.25, 0.22, 0.13, 0.10])
        amounts = np.round(np.exp(rng.normal(7.0, 1.1, size=n)), 2)

        failed = rng.random(n) < BASE_FAILURE_RATE
        status = np.where(failed, "failed", "success")
        code_idx = {
            m: rng.integers(0, len(codes), size=n)
            for m, codes in FAILURE_CODES_BY_METHOD.items()
        }
        failure_code = [
            FAILURE_CODES_BY_METHOD[m][code_idx[m][i]] if f else None
            for i, (m, f) in enumerate(zip(methods, failed, strict=True))
        ]
        latency = np.clip(rng.normal(0, 150, size=n) + np.where(failed, 900, 320), 80, None)
        latency = latency.astype(int)
        merchants = rng.integers(1, 21, size=n)
        geos = rng.choice(GEO_REGIONS, size=n)

        txn_ids = [f"txn_{self.seed:04d}_{day_index:02d}_{i:06d}" for i in range(n)]
        return list(
            zip(
                txn_ids, ts_py, amounts.tolist(), ["INR"] * n, methods.tolist(),
                list(card_network), banks.tolist(), status.tolist(), failure_code,
                latency.tolist(), [f"mch_{m:03d}" for m in merchants], geos.tolist(),
                strict=True,
            )
        )

    def generate(self) -> duckdb.DuckDBPyConnection:
        """Create a fresh in-memory DuckDB with schema + baseline transactions."""
        con = duckdb.connect()
        con.execute(SCHEMA)
        data = []
        for day in range(self.window.baseline_days + 1):
            data.extend(self._generate_rows(day))
        # executemany is ~1k rows/s; COPY from CSV is ~1M rows/s
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
            csv.writer(fh).writerows(data)
            path = fh.name
        try:
            # COPY takes no bound parameters for the file path, so the path is
            # interpolated: reject quotes to keep TMPDIR-style injection out.
            if "'" in path:
                raise ValueError(f"unsafe temp path for COPY: {path!r}")
            con.execute(
                f"COPY transactions FROM '{path}' (FORMAT CSV, NULLSTR '')"
            )
        finally:
            os.unlink(path)
        return con
