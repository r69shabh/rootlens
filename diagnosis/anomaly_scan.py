"""Stage 1: deterministic segmented anomaly scan. No LLM involved."""

from __future__ import annotations

from dataclasses import dataclass

from duckdb import DuckDBPyConnection

SUCCESS_RATE = "AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END)"

# Shared scan thresholds. The scan and the onset estimator must agree:
# a slice the scan ignores cannot trigger an onset point. Centralizing
# prevents the two from drifting apart.
SCAN_MIN_VOLUME = 15        # ignore slices with < N txns in the current window
SCAN_MIN_DROP = 0.08        # ignore slices with smaller success-rate drops
SCAN_MIN_Z = 4.0            # two-proportion z-test threshold (~40 slices tested)
ONSET_MIN_DROP = 0.10       # a window hour is "degraded" if it falls this much
ONSET_MIN_VOLUME = 20       # ...and carries at least this many txns

DIMENSIONS = {
    "issuer_bank": "issuer_bank",
    "card_network": "card_network",
    "payment_method": "payment_method",
    "amount_bucket": """
        CASE WHEN amount < 500 THEN '<500'
             WHEN amount < 2000 THEN '500-2k'
             WHEN amount < 10000 THEN '2k-10k'
             ELSE '>10k' END""",
    "geo_region": "geo_region",
    "merchant_id": "merchant_id",
}


@dataclass(frozen=True)
class AnomalousSegment:
    dimension: str
    value: str
    baseline_rate: float
    current_rate: float
    current_volume: int
    volume_share: float
    drop: float                 # baseline_rate - current_rate (positive = degraded)
    impact: float               # |drop| * volume_share

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "baseline_success_rate": round(self.baseline_rate, 4),
            "current_success_rate": round(self.current_rate, 4),
            "current_volume": self.current_volume,
            "volume_share": round(self.volume_share, 4),
            "drop": round(self.drop, 4),
            "impact": round(self.impact, 4),
        }


def _significant(base_rate: float, base_n: int, cur_rate: float, cur_n: int,
                 min_z: float) -> bool:
    """Two-proportion z-test on success rates (current vs baseline)."""
    pooled = (base_rate * base_n + cur_rate * cur_n) / (base_n + cur_n)
    se = (pooled * (1 - pooled) * (1 / cur_n + 1 / base_n)) ** 0.5
    if se <= 0:
        return False
    return abs(base_rate - cur_rate) / se >= min_z


def _segment_expr(dim: str) -> str:
    if dim not in DIMENSIONS:
        raise ValueError(f"unknown dimension {dim!r}; known: {sorted(DIMENSIONS)}")
    return DIMENSIONS[dim]


def scan(con: DuckDBPyConnection, current_start, current_end, baseline_start, baseline_end,
         min_volume: int = SCAN_MIN_VOLUME, min_drop: float = SCAN_MIN_DROP,
         min_z: float = SCAN_MIN_Z) -> list[AnomalousSegment]:
    """Compare every dimension slice current-vs-baseline; rank by impact.

    Only slices with meaningful volume, a material drop, AND statistical
    significance (two-proportion z-test, conservative threshold because we test
    ~40 slices per scan) are reported. This keeps the false-positive rate on
    healthy and benign-spike windows near zero without missing real faults.
    """
    total = con.execute(
        "SELECT COUNT(*) FROM transactions WHERE ts >= ? AND ts < ?",
        [current_start, current_end],
    ).fetchone()[0]
    if total == 0:
        return []

    segments: list[AnomalousSegment] = []
    for dim, expr in DIMENSIONS.items():
        rows = con.execute(
            f"""
            WITH cur AS (
                SELECT ({expr}) AS seg, COUNT(*) AS n, {SUCCESS_RATE} AS sr
                FROM transactions WHERE ts >= ? AND ts < ?
                GROUP BY 1
            ),
            base AS (
                SELECT ({expr}) AS seg, COUNT(*) AS n, {SUCCESS_RATE} AS sr
                FROM transactions WHERE ts >= ? AND ts < ?
                GROUP BY 1
            )
            SELECT COALESCE(cur.seg, 'NULL'), cur.n, cur.sr, base.sr, base.n
            FROM cur LEFT JOIN base USING (seg)
            """,
            [current_start, current_end, baseline_start, baseline_end],
        ).fetchall()
        for seg, n, cur_sr, base_sr, base_n in rows:
            if base_sr is None or n < min_volume:
                continue
            drop = base_sr - cur_sr
            if drop < min_drop:
                continue
            if not _significant(base_sr, base_n, cur_sr, n, min_z):
                continue
            segments.append(
                AnomalousSegment(
                    dimension=dim, value=str(seg),
                    baseline_rate=base_sr, current_rate=cur_sr,
                    current_volume=n, volume_share=n / total,
                    drop=drop, impact=drop * n / total,
                )
            )
    segments = _prune_explained_parents(segments)
    # Rank by drop first, impact as volume tiebreak: impact alone buries small-volume
    # but decisive slices (e.g. a >10k bucket at 70% failure) beneath large diffuse
    # slices whose anomaly is fully explained by a concentrated child (see prune).
    segments.sort(key=lambda s: (s.drop, s.impact), reverse=True)
    return segments


def _prune_explained_parents(segments: list[AnomalousSegment]) -> list[AnomalousSegment]:
    """Remove coarse slices whose anomaly is fully explained by a specific child.

    E.g. payment_method=card degrades because visa alone degraded; the card-level
    segment adds no diagnostic value and would outrank the real cause by volume.
    """
    # hierarchy: payment_method=card is explained by a card_network slice
    parents_to_drop = set()
    by_dim = {}
    for s in segments:
        by_dim.setdefault(s.dimension, []).append(s)
    card = next((s for s in by_dim.get("payment_method", []) if s.value == "card"), None)
    nets = by_dim.get("card_network", [])
    if card and nets and any(n.drop >= card.drop * 0.9 for n in nets):
        parents_to_drop.add(("payment_method", "card"))
    return [s for s in segments if (s.dimension, s.value) not in parents_to_drop]


def hourly_success_rate(con, start, end, filters: dict | None = None) -> list[dict]:
    """Hourly success rate in [start, end); used for onset estimation."""
    where = ["ts >= ?", "ts < ?"]
    params: list = [start, end]
    for col, val in (filters or {}).items():
        where.append(f"{col} = ?")
        params.append(val)
    rows = con.execute(
        f"""
        SELECT date_trunc('hour', ts) AS hour,
               COUNT(*) AS n, {SUCCESS_RATE} AS sr
        FROM transactions WHERE {' AND '.join(where)}
        GROUP BY 1 ORDER BY 1
        """,
        params,
    ).fetchall()
    return [{"hour": str(h), "volume": n, "success_rate": round(sr, 4)} for h, n, sr in rows]


def estimate_onset(con, current_start, current_end, baseline_rate: float,
                   filters: dict | None = None) -> str | None:
    """First hour in the current window whose success rate falls well below baseline."""
    points = hourly_success_rate(con, current_start, current_end, filters)
    for p in points:
        if (p["success_rate"] < baseline_rate - ONSET_MIN_DROP
                and p["volume"] >= ONSET_MIN_VOLUME):
            return p["hour"]
    return None
