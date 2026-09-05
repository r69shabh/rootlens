"""Model leaderboard: accuracy x latency x cost, per agent, on the same scenario
set (architecture section 5). The rule-based baseline anchors the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1M tokens (published list prices; update before quoting costs publicly).
PROVIDER_PRICES: dict[str, dict[str, float]] = {
    "rule-baseline": {"input": 0.0, "output": 0.0},
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic:claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "gemini": {"input": 0.10, "output": 0.40},
}


@dataclass
class RunRecord:
    scenario_id: str
    tier: str
    agent: str
    status: str
    correct: bool
    partial: bool
    inconclusive: bool
    false_positive: bool
    latency_minutes: float
    llm_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False

    @property
    def est_cost_usd(self) -> float:
        prices = PROVIDER_PRICES.get(self.agent, {"input": 0.0, "output": 0.0})
        return (
            self.input_tokens / 1e6 * prices["input"] + self.output_tokens / 1e6 * prices["output"]
        )


@dataclass
class AgentStanding:
    agent: str
    runs: int = 0
    correct: int = 0
    inconclusive: int = 0
    false_positives: int = 0
    latency_minutes: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    per_tier: dict = field(default_factory=dict)
    any_estimated_tokens: bool = False

    @property
    def accuracy(self) -> float:
        return self.correct / self.runs if self.runs else 0.0

    def summary(self) -> str:
        tok = f", tokens {self.input_tokens + self.output_tokens:,}"
        if self.any_estimated_tokens:
            tok += " (partly estimated)"
        return (
            f"{self.agent:32s} acc {self.accuracy:5.0%} ({self.correct}/{self.runs}) "
            f"latency {self.latency_minutes / max(self.runs, 1):6.2f} min/run "
            f"cost ${self.est_cost_usd:.4f}{tok}"
        )


class Leaderboard:
    def __init__(self) -> None:
        self.records: list[RunRecord] = []

    def add(self, record: RunRecord) -> None:
        self.records.append(record)

    def standings(self) -> list[AgentStanding]:
        by_agent: dict[str, AgentStanding] = {}
        for r in self.records:
            st = by_agent.setdefault(r.agent, AgentStanding(agent=r.agent))
            st.runs += 1
            st.correct += int(r.correct)
            st.inconclusive += int(r.inconclusive)
            st.false_positives += int(r.false_positive)
            st.latency_minutes += r.latency_minutes
            st.llm_calls += r.llm_calls
            st.input_tokens += r.input_tokens
            st.output_tokens += r.output_tokens
            st.est_cost_usd += r.est_cost_usd
            st.any_estimated_tokens = st.any_estimated_tokens or r.tokens_estimated
            tier = st.per_tier.setdefault(r.tier, {"total": 0, "correct": 0})
            tier["total"] += 1
            tier["correct"] += int(r.correct)
        # rank: accuracy first, then cheaper, then faster
        return sorted(
            by_agent.values(), key=lambda s: (-s.accuracy, s.est_cost_usd, s.latency_minutes)
        )

    def to_markdown(self) -> str:
        lines = [
            "# RootLens diagnosis leaderboard",
            "",
            "Same scenario set per agent. Rank: accuracy, then cost, then latency.",
            "",
            "| Agent | Accuracy | Inconclusive | False pos. | Latency (min/run) "
            "| LLM calls | Est. cost/run ($) |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in self.standings():
            cost_per_run = s.est_cost_usd / max(s.runs, 1)
            lines.append(
                f"| {s.agent} | {s.accuracy:.0%} ({s.correct}/{s.runs}) "
                f"| {s.inconclusive} | {s.false_positives} "
                f"| {s.latency_minutes / max(s.runs, 1):.2f} "
                f"| {s.llm_calls} | {cost_per_run:.4f} |"
            )
        tiers = sorted({r.tier for r in self.records})
        lines += [
            "",
            "## Per-tier accuracy",
            "",
            "| Agent | " + " | ".join(tiers) + " |",
            "|---|" + "---|" * len(tiers),
        ]
        for s in self.standings():
            cells = []
            for t in tiers:
                d = s.per_tier.get(t, {"total": 0, "correct": 0})
                acc = d["correct"] / d["total"] if d["total"] else float("nan")
                cells.append(f"{acc:.0%} ({d['correct']}/{d['total']})")
            lines.append(f"| {s.agent} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"
