"""Per-tier accuracy scoring against ground truth. Never blended into one number."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from diagnosis.agent import DiagnosisResult


def _normalize(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _family_tokens(label: str) -> set[str]:
    """Split a normalized label into exact tokens (family + qualifiers).

    Substring matching ("outage" in "bank_outage:...") inflates accuracy;
    token membership does not.
    """
    return set(re.split(r"[^a-z0-9_]+", _normalize(label))) - {""}


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
    pred_tokens = _family_tokens(predicted)
    # Match on family tokens (the part of each label before ':'), so compound
    # predictions like "compound:bank_outage+rule_trigger" score correct when every
    # injected fault family is covered.
    families = [lbl.split(":")[0] for lbl in expected]
    if len(families) > 1:
        correct = all(f in pred_tokens for f in families)
    else:
        fam = families[0]
        correct = fam in pred_tokens
    return {
        "scenario_id": ground_truth["scenario_id"], "tier": ground_truth["difficulty_tier"],
        "correct": correct, "partial": (not correct) and any(f in pred_tokens for f in families),
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

    def wilson_95ci(self) -> tuple[float, float]:
        """Wilson 95% confidence interval on the accuracy proportion.

        Honest per-tier numbers need this: with 8/8 = 100% the interval is
        [0.63, 1.00], so the README's "100%" is misleading on small N.
        """
        return _wilson_95ci(self.correct, self.total)

    def summary(self) -> str:
        if self.total:
            lo, hi = self.wilson_95ci()
            return (f"[{self.tier}] accuracy {self.correct}/{self.total} = {self.accuracy:.0%} "
                    f"(95% CI {lo:.0%}-{hi:.0%}), "
                    f"inconclusive {self.inconclusive}, false_positives {self.false_positives}")
        return (f"[{self.tier}] no runs, "
                f"inconclusive {self.inconclusive}, false_positives {self.false_positives}")


def _wilson_95ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well-behaved at 0/0 and 1/1, unlike the
    normal approximation. Standard reference implementation."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_batch(results: list[tuple[DiagnosisResult, dict]]) -> dict[str, TierReport]:
    reports: dict[str, TierReport] = {}
    for result, gt in results:
        tier = gt["difficulty_tier"]
        rep = reports.setdefault(tier, TierReport(tier=tier))
        s = score_result(result, gt)
        rep.total += 1
        rep.correct += int(s["correct"])
        rep.inconclusive += int(s["inconclusive"])
        rep.false_positives += int(s.get("false_positive", False))
        rep.details.append(s)
    return reports
