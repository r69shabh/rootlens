"""Grounded priors: historical fault frequencies, given to the LLM as structured context."""

from __future__ import annotations

# Historical frequency of root-cause families across past incidents.
# Deliberately explicit data, not internal LLM assumptions.
FAULT_PRIORS: dict[str, float] = {
    "bank_outage": 0.28,
    "network_degradation": 0.22,
    "high_ticket_rule": 0.15,
    "retry_storm": 0.10,
    "checkout_funnel_break": 0.08,
    "settlement_delay": 0.07,
    "other": 0.10,
}


def priors_context() -> str:
    lines = [f"- {name}: {freq:.0%}" for name, freq in sorted(
        FAULT_PRIORS.items(), key=lambda kv: -kv[1])]
    return "Historical fault-rate priors (frequency of past root causes):\n" + "\n".join(lines)
