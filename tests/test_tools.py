"""Tools: whitelist enforcement, SQL parameterization, correct metrics."""

import pytest

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig
from diagnosis.evidence import EvidenceStore
from diagnosis.tools import DiagnosisTools, ToolError

WC = WindowConfig(start=DEFAULT_WINDOW_START)


def _tools(con):
    return DiagnosisTools(con, EvidenceStore())


def test_rejects_disallowed_filter_column(healthy_con):
    t = _tools(healthy_con)
    with pytest.raises(ToolError):
        t.query_transactions(filters={"txn_id": "x"})
    with pytest.raises(ToolError):
        t.query_transactions(filters={"1=1; DROP TABLE": "x"})


def test_rejects_disallowed_metric_and_group_by(healthy_con):
    t = _tools(healthy_con)
    with pytest.raises(ToolError):
        t.query_transactions(metrics=["amount"])
    with pytest.raises(ToolError):
        t.query_transactions(group_by=["amount"])
    with pytest.raises(ToolError):
        t.timeseries(metric="banana", granularity="hour", start=None, end=None)


def test_unknown_tool_raises(healthy_con):
    t = _tools(healthy_con)
    with pytest.raises(ToolError):
        t.dispatch("delete_table", {})


def test_query_by_issuer_bank_filters_correctly(healthy_con):
    t = _tools(healthy_con)
    rows = t.query_transactions(
        filters={"issuer_bank": "ICICI"}, group_by=["status"],
        metrics=["count"], start=WC.start, end=WC.end,
    )
    total_icici = healthy_con.execute(
        "SELECT COUNT(*) FROM transactions WHERE issuer_bank='ICICI' AND ts >= ? AND ts < ?",
        [WC.start, WC.end],
    ).fetchone()[0]
    assert sum(r["count"] for r in rows) == total_icici


def test_timeseries_and_baseline_compare(healthy_con):
    t = _tools(healthy_con)
    ts = t.timeseries("success_rate", "hour", WC.current_window_start, WC.current_window_end)
    assert len(ts) >= 2
    z = t.baseline_compare("success_rate", WC.current_window_start, WC.current_window_end,
                           WC.start, WC.current_window_start)
    assert abs(z["z_score"]) < 3  # healthy data: no big deviation


def test_every_call_is_logged_to_evidence_store(healthy_con):
    store = EvidenceStore()
    t = DiagnosisTools(healthy_con, store)
    t.query_transactions(filters={"issuer_bank": "SBI"}, start=WC.start, end=WC.end)
    t.timeseries("count", "hour", WC.current_window_start, WC.current_window_end)
    assert len(store.entries) == 2
    assert store.entries[0].call_id == "call_001"
    assert store.entries[0].args["filters"] == {"issuer_bank": "SBI"}
    assert store.get("call_002") is not None
