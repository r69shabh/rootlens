"""Tools: whitelist enforcement, SQL parameterization, correct metrics."""

from datetime import timedelta

import pytest

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig
from diagnosis.evidence import EvidenceStore
from diagnosis.tools import MAX_RESULT_ROWS, TRUNCATION_KEY, DiagnosisTools, ToolError

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


def test_result_capped_with_truncation_flag(healthy_con):
    # group_by issuer_bank x payment_method x card_network x status x failure_code
    # yields more rows than MAX_RESULT_ROWS on the healthy dataset.
    t = _tools(healthy_con)
    rows = t.query_transactions(
        group_by=["issuer_bank", "payment_method", "status", "failure_code"],
        metrics=["count"], start=WC.start, end=WC.end,
    )
    assert len(rows) <= MAX_RESULT_ROWS + 1  # +1 for the truncation marker
    flagged = [r for r in rows if r.get(TRUNCATION_KEY)]
    if len(flagged) == 0:
        # If the healthy dataset actually fits under cap, we just confirm the
        # cap is generous enough — the truncation path is unit-tested below.
        assert len(rows) <= MAX_RESULT_ROWS


def test_truncation_marker_appears_on_exploding_groupby(monkeypatch, healthy_con):
    # force a tiny cap so a single 5-column groupby triggers truncation
    import diagnosis.tools as tools_mod
    monkeypatch.setattr(tools_mod, "MAX_RESULT_ROWS", 5)
    t = _tools(healthy_con)
    rows = t.query_transactions(
        group_by=["issuer_bank", "payment_method", "status", "failure_code"],
        metrics=["count"], start=WC.start, end=WC.end,
    )
    assert len(rows) == 6  # 5 data rows + 1 truncation marker
    assert rows[-1].get(TRUNCATION_KEY) is True
    assert rows[-1]["rows_returned"] == 5


def test_baseline_compare_marks_unreliable_with_one_hour(healthy_con):
    # baseline window is 5 days; we craft a 30-min filter so the resulting
    # baseline has just 1 hourly bucket -> std=0, z-score must be flagged
    # unreliable rather than reported as a huge number.
    tiny_start = WC.start
    tiny_end = WC.start + timedelta(minutes=30)
    t = _tools(healthy_con)
    out = t.baseline_compare("success_rate", WC.current_window_start, WC.current_window_end,
                             tiny_start, tiny_end)
    assert out["baseline"]["n_hours"] == 1
    assert out.get("z_score_reliable") is False
    # magnitude sanity: with the new std floor the score is bounded, not 1e6
    assert abs(out["z_score"]) < 1000
