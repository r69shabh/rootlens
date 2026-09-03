"""Eval harness scoring."""

from diagnosis.agent import DiagnosisResult
from eval.harness import score_batch, score_result


def _gt(tier="clean", labels=("bank_outage:ICICI",), types=("bank_outage",), sid="s1"):
    return {
        "scenario_id": sid,
        "difficulty_tier": tier,
        "expected_labels": list(labels),
        "expected_fault_types": list(types),
    }


def test_exact_match_scores_correct():
    r = DiagnosisResult(status="verdict", root_cause="bank_outage:ICICI", confidence=0.9)
    assert score_result(r, _gt())["correct"] is True


def test_wrong_cause_scores_incorrect():
    r = DiagnosisResult(status="verdict", root_cause="network_degradation:visa", confidence=0.9)
    s = score_result(r, _gt())
    assert s["correct"] is False


def test_inconclusive_is_scored_honestly():
    r = DiagnosisResult(status="inconclusive", missing="need more data")
    s = score_result(r, _gt())
    assert s["correct"] is False and s["inconclusive"] is True


def test_compound_requires_all_fault_types():
    gt = _gt(
        tier="compound",
        labels=("bank_outage:ICICI", "rule_trigger:10000"),
        types=("bank_outage", "high_ticket_rule"),
    )
    full = DiagnosisResult(
        status="verdict", root_cause="compound:bank_outage+rule_trigger", confidence=0.8
    )
    assert score_result(full, gt)["correct"] is True
    half = DiagnosisResult(status="verdict", root_cause="bank_outage:ICICI", confidence=0.8)
    assert score_result(half, gt)["correct"] is False
    assert score_result(half, gt)["partial"] is True


def test_batch_reports_per_tier():
    results = [
        (DiagnosisResult(status="verdict", root_cause="bank_outage:ICICI"), _gt(sid="a")),
        (DiagnosisResult(status="verdict", root_cause="wrong:x"), _gt(sid="b")),
        (
            DiagnosisResult(status="verdict", root_cause="bank_outage:ICICI"),
            _gt(tier="compound", sid="c"),
        ),
    ]
    reports = score_batch(results)
    assert reports["clean"].accuracy == 0.5
    assert reports["compound"].accuracy == 1.0
