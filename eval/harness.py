"""Per-tier accuracy scoring against ground truth. Never blended into one number."""

from __future__ import annotations

from dataclasses import dataclass, field

from diagnosis.agent import DiagnosisResult


def _normalize(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def score_result(result: DiagnosisResult, ground_truth: dict) -> dict:
    expected = [_normalize(x) for x in ground_truth["expected_labels"]]
    if not expected:
        # control scenario (healthy / benign spike): correct means NOT raising a verdict
        correct = result.status != "verdict"
        return {
            "scenario_id": ground_truth["scenario_id"], "tier": ground_truth["difficulty_tier"],
            "correct": correct, "partial": False, "inconclusive": result.status == "inconclusive",
            "predicted": result.root_cause if result.status == "verdict" else None,
            "expected": [], "false_positive": result.status == "verdict",
        }
    if result.status != "verdict":
        return {
            "scenario_id": ground_truth["scenario_id"], "tier": ground_truth["difficulty_tier"],
            "correct": False, "partial": False, "inconclusive": result.status == "inconclusive",
            "predicted": None, "expected": expected,
        }
    predicted = _normalize(result.root_cause or "")
    # Match on family tokens (the part of each label before ':'), so compound
    # predictions like "compound:bank_outage+rule_trigger" score correct when every
    # injected fault family is covered.
    families = [lbl.split(":")[0] for lbl in expected]
    if len(families) > 1:
        correct = all(f in predicted for f in families)
    else:
        fam = families[0]
        correct = fam in predicted
    return {
        "scenario_id": ground_truth["scenario_id"], "tier": ground_truth["difficulty_tier"],
        "correct": correct, "partial": (not correct) and any(f in predicted for f in families),
        "inconclusive": False, "predicted": result.root_cause, "expected": expected,
        "confidence": result.confidence, "rounds_used": result.rounds_used,
    }


@dataclass
class TierReport:
    tier: str
    total: int = 0
    correct: int = 0
    inconclusive: int = 0
    false_positives: int = 0  # healthy scenarios flagged as anomalous
    details: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> str:
        return (f"[{self.tier}] accuracy {self.correct}/{self.total} = {self.accuracy:.0%}, "
                f"inconclusive {self.inconclusive}, false_positives {self.false_positives}")


def score_batch(results: list[tuple[DiagnosisResult, dict]]) -> dict[str, TierReport]:
    reports: dict[str, TierReport] = {}
    for result, gt in results:
        tier = gt["difficulty_tier"]
        rep = reports.setdefault(tier, TierReport(tier=tier))
        s = score_result(result, gt)
        rep.total += 1
        rep.correct += int(s["correct"])
        rep.inconclusive += int(s["inconclusive"])
        rep.false_positives += int(gt["scenario_id"] == "healthy" and s["predicted"] is not None)
        rep.details.append(s)
    return reports
