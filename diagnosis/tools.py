"""The only path from the agent to data. Read-only, whitelisted, fully parameterized."""

from __future__ import annotations

import time

from duckdb import DuckDBPyConnection

from diagnosis.evidence import EvidenceStore

# Whitelists: the agent can never touch arbitrary columns or SQL.
FILTERABLE_COLUMNS = {
    "issuer_bank", "card_network", "payment_method", "amount_bucket",
    "geo_region", "merchant_id", "status", "failure_code",
}
GROUPABLE_COLUMNS = FILTERABLE_COLUMNS | {"ts_hour"}
METRICS = {"count", "success_rate", "failure_rate", "avg_amount", "p95_latency"}

AMOUNT_BUCKET_SQL = """
CASE WHEN amount < 500 THEN '<500'
     WHEN amount < 2000 THEN '500-2k'
     WHEN amount < 10000 THEN '2k-10k'
     ELSE '>10k' END"""

_COLUMN_SQL = {
    "issuer_bank": "issuer_bank",
    "card_network": "card_network",
    "payment_method": "payment_method",
    "geo_region": "geo_region",
    "merchant_id": "merchant_id",
    "status": "status",
    "failure_code": "failure_code",
    "amount_bucket": AMOUNT_BUCKET_SQL,
    "ts_hour": "date_trunc('hour', ts)",
}

_METRIC_SQL = {
    "count": "COUNT(*)",
    "success_rate": "AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END)",
    "failure_rate": "AVG(CASE WHEN status = 'failed' THEN 1.0 ELSE 0.0 END)",
    "avg_amount": "AVG(amount)",
    "p95_latency": "quantile_cont(gateway_latency_ms, 0.95)",
}


class ToolError(ValueError):
    """Raised when the agent asks for something outside the whitelist."""


def _validate_filters(filters: dict) -> dict:
    clean = {}
    for col, val in (filters or {}).items():
        if col not in FILTERABLE_COLUMNS:
            raise ToolError(
                f"filter column {col!r} not allowed; allowed: {sorted(FILTERABLE_COLUMNS)}"
            )
        clean[col] = val
    return clean


def _where_clause(filters: dict, start=None, end=None) -> tuple[str, list]:
    conds, params = [], []
    if start is not None:
        conds.append("ts >= ?")
        params.append(start)
    if end is not None:
        conds.append("ts < ?")
        params.append(end)
    for col, val in filters.items():
        if col == "amount_bucket":
            conds.append(f"({AMOUNT_BUCKET_SQL}) = ?")
        else:
            conds.append(f"{col} = ?")
        params.append(val)
    return (" AND ".join(conds)) or "TRUE", params


class DiagnosisTools:
    """Binds a DuckDB connection + evidence store into callable tools for the agent."""

    def __init__(self, con: DuckDBPyConnection, store: EvidenceStore) -> None:
        self.con = con
        self.store = store

    def _run(self, tool: str, args: dict, sql: str, params: list) -> list[dict]:
        t0 = time.perf_counter()
        rows = self.con.execute(sql, params).fetchall()
        cols = [d[0] for d in self.con.description]
        result = [dict(zip(cols, r, strict=True)) for r in rows]
        self.store.log(tool, args, result, (time.perf_counter() - t0) * 1000)
        return result

    def query_transactions(self, filters: dict | None = None, group_by: list[str] | None = None,
                           metrics: list[str] | None = None, start=None, end=None) -> list[dict]:
        """Aggregate transactions with whitelisted filters/group_bys/metrics."""
        args = {"filters": filters, "group_by": group_by, "metrics": metrics,
                "start": str(start) if start else None, "end": str(end) if end else None}
        filters = _validate_filters(filters or {})
        group_by = group_by or []
        metrics = metrics or ["count", "success_rate"]
        for m in metrics:
            if m not in METRICS:
                raise ToolError(f"metric {m!r} not allowed; allowed: {sorted(METRICS)}")
        for g in group_by:
            if g not in GROUPABLE_COLUMNS:
                raise ToolError(f"group_by {g!r} not allowed; allowed: {sorted(GROUPABLE_COLUMNS)}")
        where, params = _where_clause(filters, start, end)
        select_parts = [_COLUMN_SQL[g] + f" AS {g}" for g in group_by]
        select_parts += [f"{_METRIC_SQL[m]} AS {m}" for m in metrics]
        sql = f"SELECT {', '.join(select_parts)} FROM transactions WHERE {where}"
        if group_by:
            sql += " GROUP BY " + ", ".join(str(i + 1) for i in range(len(group_by)))
            sql += " ORDER BY " + ", ".join(f"{m} DESC" for m in metrics if m in ("count",))
        return self._run("query_transactions", args, sql, params)

    def timeseries(self, metric: str, granularity: str, start, end,
                   filters: dict | None = None) -> list[dict]:
        """Metric over time buckets, for onset alignment."""
        args = {"metric": metric, "granularity": granularity, "start": str(start),
                "end": str(end), "filters": filters}
        if metric not in METRICS:
            raise ToolError(f"metric {metric!r} not allowed; allowed: {sorted(METRICS)}")
        if granularity not in {"hour", "minute", "day"}:
            raise ToolError("granularity must be hour | minute | day")
        filters = _validate_filters(filters or {})
        where, params = _where_clause(filters, start, end)
        sql = f"""
            SELECT date_trunc('{granularity}', ts) AS bucket, {_METRIC_SQL[metric]} AS value
            FROM transactions WHERE {where}
            GROUP BY 1 ORDER BY 1
        """
        return self._run("timeseries", args, sql, params)

    def compare_segments(self, dim_a: str, dim_b: str, start, end) -> list[dict]:
        """Concentrated-vs-diffuse test: does a failure signal in dim_a also
        appear across dim_b slices? (e.g. one issuer vs. whole network)"""
        args = {"dim_a": dim_a, "dim_b": dim_b, "start": str(start), "end": str(end)}
        for d in (dim_a, dim_b):
            if d not in GROUPABLE_COLUMNS:
                raise ToolError(
                    f"dimension {d!r} not allowed; allowed: {sorted(GROUPABLE_COLUMNS)}"
                )
        sql = f"""
            SELECT {_COLUMN_SQL[dim_a]} AS {dim_a},
                   {_COLUMN_SQL[dim_b]} AS {dim_b},
                   COUNT(*) AS count,
                   AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate
            FROM transactions
            WHERE ts >= ? AND ts < ?
            GROUP BY 1, 2 ORDER BY 1, 2
        """
        return self._run("compare_segments", args, sql, [start, end])

    def baseline_compare(self, metric: str, current_start, current_end,
                         baseline_start, baseline_end, filters: dict | None = None) -> dict:
        """Z-score of the current-window metric against the baseline window."""
        args = {"metric": metric, "current": [str(current_start), str(current_end)],
                "baseline": [str(baseline_start), str(baseline_end)], "filters": filters}
        if metric not in METRICS:
            raise ToolError(f"metric {metric!r} not allowed; allowed: {sorted(METRICS)}")
        filters = _validate_filters(filters or {})
        expr = _METRIC_SQL[metric]
        out: dict = {"metric": metric}
        for name, (s, e) in {"baseline": (baseline_start, baseline_end),
                             "current": (current_start, current_end)}.items():
            where, params = _where_clause(filters, s, e)
            # hourly samples -> distribution for the z-score, not a single aggregate
            rows = self.con.execute(
                f"""
                SELECT date_trunc('hour', ts) AS h, {expr} AS v
                FROM transactions WHERE {where}
                GROUP BY 1
                """,
                params,
            ).fetchall()
            vals = [r[1] for r in rows if r[1] is not None]
            if not vals:
                raise ToolError(f"no data for {name} window")
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            out[name] = {"mean": mean, "std": var ** 0.5, "n_hours": len(vals),
                         "latest": vals[-1] if name == "current" else None}
        std = max(out["baseline"]["std"], 1e-9)
        out["z_score"] = (out["current"]["mean"] - out["baseline"]["mean"]) / std
        self.store.log("baseline_compare", args, out, 0.0)
        return out

    def dispatch(self, tool: str, args: dict):
        fn = getattr(self, tool, None)
        if fn is None or tool.startswith("_"):
            raise ToolError(f"unknown tool {tool!r}")
        return fn(**args)
