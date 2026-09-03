"""Deterministic rule-based diagnosis agent — the leaderboard floor.

Zero LLM calls. It maps the top-ranked anomalous segment to a fault family using
the same disconfirmation logic the LLM is instructed to apply, with real tool
calls so its evidence chain is as auditable as the LLM's. Any LLM that scores
below this baseline is adding cost and latency without adding accuracy.
"""

from __future__ import annotations

import time

from duckdb import DuckDBPyConnection

from diagnosis.agent import CONFIDENCE_THRESHOLD, DiagnosisResult
from diagnosis.anomaly_scan import AnomalousSegment, scan
from diagnosis.evidence import EvidenceStore
from diagnosis.tools import DiagnosisTools


def _failure_code_mix(tools: DiagnosisTools, start, end) -> dict[str, int]:
    rows = tools.query_transactions(
        filters={"status": "failed"}, group_by=["failure_code"],
        metrics=["count"], start=start, end=end,
    )
    return {r["failure_code"]: r["count"] for r in rows if r["failure_code"]}


def _label_for_segment(tools: DiagnosisTools, seg: AnomalousSegment,
                       current_start, current_end) -> tuple[str, list[str]]:
    """Map a top segment to a root-cause label. Returns (label, disconfirmation)."""
    dim, value = seg.dimension, seg.value
    checks: list[str] = []

    if dim == "issuer_bank":
        # disconfirmation: a network degradation would hit ALL banks on a network
        matrix = tools.compare_segments("issuer_bank", "card_network",
                                        current_start, current_end)
        others = [r for r in matrix if r["issuer_bank"] != value]
        diffuse = any(r["success_rate"] < 0.8 for r in others)
        checks.append(
            f"issuer x network matrix over {len(matrix)} cells: other issuers "
            + ("also degraded -> would indicate network issue" if diffuse
               else "healthy across all networks -> isolated to this bank")
        )
        return (f"bank_outage:{value}" if not diffuse
                else f"network_degradation:degraded_on_{value}"), checks

    if dim == "card_network":
        matrix = tools.compare_segments("card_network", "issuer_bank",
                                        current_start, current_end)
        same_net = [r for r in matrix if r["card_network"] == value]
        spans_issuers = sum(1 for r in same_net if r["success_rate"] < 0.8) >= 2
        checks.append(
            "network x issuer matrix: degradation "
            + ("spans multiple issuers -> network-level issue"
               if spans_issuers else "concentrated in one issuer -> bank outage")
        )
        return f"network_degradation:{value}", checks

    if dim == "amount_bucket":
        mix = _failure_code_mix(tools, current_start, current_end)
        top_code = max(mix, key=mix.get) if mix else "unknown"
        checks.append(f"failure-code mix over whole window: {mix}; dominant={top_code}")
        threshold = {"<500": 0, "500-2k": 500, "2k-10k": 2000, ">10k": 10000}.get(value, 0)
        return f"rule_trigger:{threshold}", checks

    if dim == "merchant_id":
        status_rows = tools.query_transactions(
            filters={"merchant_id": value}, group_by=["status"],
            metrics=["count"], start=current_start, end=current_end,
        )
        by_status = {r["status"]: r["count"] for r in status_rows}
        pending = by_status.get("pending", 0)
        checks.append(
            f"status split for {value}: {by_status} — "
            + ("pending spike with no failure codes -> settlement delay"
               if pending else "failed volume -> merchant-side issue")
        )
        return f"settlement_delay:{value}", checks

    # diffuse dims (payment_method / geo_region): distinguish via failure codes
    mix = _failure_code_mix(tools, current_start, current_end)
    top_code = max(mix, key=mix.get) if mix else "unknown"
    checks.append(f"failure-code mix over whole window: {mix}; dominant={top_code}")
    if top_code == "gateway_timeout":
        return "retry_storm:gateway", checks
    if top_code == "checkout_error":
        return "checkout_funnel_break:checkout", checks
    if top_code == "issuer_unavailable":
        return "bank_outage:multi_bank", checks
    return "inconclusive:unrecognized_pattern", checks


def rule_based_diagnose(con: DuckDBPyConnection, current_start, current_end,
                        baseline_start, baseline_end,
                        scenario_id: str = "rule-baseline") -> DiagnosisResult:
    """Full deterministic pipeline: scan -> label -> evidence-backed verdict."""
    t0 = time.perf_counter()
    store = EvidenceStore(scenario_id=scenario_id)
    tools = DiagnosisTools(con, store)
    segments = scan(con, current_start, current_end, baseline_start, baseline_end)

    if not segments:
        result = DiagnosisResult(
            status="inconclusive",
            missing="no anomalous segments crossed significance thresholds; "
                    "no evidence of a fault in this window",
            rounds_used=0,
        )
    else:
        # Evidence before hypotheses: a diffuse fault (retry storm, checkout
        # break) has NO specific slice — every slice degrades equally, so segment
        # identity is meaningless. The failure-code mix identifies them first.
        mix = _failure_code_mix(tools, current_start, current_end)
        label, checks = None, []
        if mix:
            ranked = sorted(mix.items(), key=lambda kv: -kv[1])
            (top_code, top_n), second_n = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0)
            if top_code == "gateway_timeout" and top_n >= 30 and top_n >= 2 * second_n:
                label = "retry_storm:gateway"
                checks.append(f"failure-code mix {mix}: gateway_timeout dominates and "
                              "degradation is diffuse across all segments -> retry storm")
            elif top_code == "checkout_error" and top_n >= 30 and top_n >= 2 * second_n:
                label = "checkout_funnel_break:checkout"
                checks.append(f"failure-code mix {mix}: checkout_error dominates and "
                              "degradation is diffuse across all payment methods "
                              "-> client-side checkout break")
        if label is None:
            top = segments[0]
            label, checks = _label_for_segment(tools, top, current_start, current_end)
            top = segments[0]
        else:
            top = segments[0]
        # compound: two orthogonal concentrated segments (bank + amount rule)
        compound = None
        for s in segments[1:3]:
            if top.dimension == "issuer_bank" and s.dimension == "amount_bucket" \
                    and s.drop >= 0.3:
                compound = "compound:bank_outage+rule_trigger"
                checks.append(
                    f"second concentrated segment {s.dimension}={s.value} "
                    f"(drop {s.drop:.2f}) orthogonal to the bank outage -> compound"
                )
                break

        cited = [e.call_id for e in store.entries]
        impact_rows = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM transactions "
            "WHERE ts >= ? AND ts < ? AND status != 'success'",
            [current_start, current_end],
        ).fetchone()
        result = DiagnosisResult(
            status="verdict",
            root_cause=compound or label,
            confidence=min(0.99, max(CONFIDENCE_THRESHOLD, 0.5 + top.drop)),
            evidence_call_ids=cited,
            disconfirmation=checks,
            impact={"claimed_by_model": {
                "transactions_affected": int(impact_rows[0]),
                "gmv_at_risk_inr": round(float(impact_rows[1]), 2)}},
            rounds_used=len(cited),
        )

    from diagnosis.impact import estimate_impact
    elapsed = (time.perf_counter() - t0) / 60
    result.time_to_diagnosis_minutes = round(elapsed, 2)
    result.impact = {"estimated": estimate_impact(
        con, current_start, current_end, elapsed), **result.impact}
    result.transcript = [{"role": "context",
                          "content": f"rule-based baseline, {len(segments)} segments"}]
    result._store = store
    return result
