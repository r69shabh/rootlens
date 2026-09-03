"""Scenario registry: build a dataset + ground-truth JSON for each eval scenario."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

from data_engine.faults import (
    BankOutageInjector,
    CheckoutFunnelBreakInjector,
    HighTicketRuleInjector,
    InjectedFault,
    MarketingSpikeInjector,
    NetworkDegradationInjector,
    RetryStormInjector,
    SettlementDelayInjector,
)
from data_engine.generator import TransactionGenerator, WindowConfig


def record_fault_events(con, faults: list[InjectedFault]) -> None:
    if not faults:
        return
    rows = [
        (f"flt_{i:03d}", f.fault_type, f.start_ts, f.end_ts,
         json.dumps(f.affected_scope, default=str), f.difficulty_tier)
        for i, f in enumerate(faults)
    ]
    con.executemany(
        "INSERT INTO fault_events VALUES (?, ?, ?, ?, ?, ?)", rows
    )


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    tier: str
    description: str
    build: object  # callable() -> (con, list[InjectedFault])

    def build_dataset(self):
        con, faults = self.build()
        record_fault_events(con, faults)
        return con, faults

    def ground_truth(self) -> dict:
        con, faults = self.build_dataset()
        con.close()
        return {
            "scenario_id": self.scenario_id,
            "difficulty_tier": self.tier,
            "expected_labels": [f.label for f in faults],
            "expected_fault_types": sorted({f.fault_type for f in faults}),
            "faults": [asdict(f) for f in faults],
        }


SEED = 42


def _base_window() -> WindowConfig:
    return WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))


def _make(gen_kwargs=None, injector=None, injectors=None, tier=None):
    gen_kwargs = gen_kwargs or {}

    def build():
        gen = TransactionGenerator(seed=SEED, **gen_kwargs)
        con = gen.generate()
        inj_list = injectors if injectors is not None else ([injector] if injector else [])
        injected = []
        for inj in inj_list:
            out = inj.inject(con)
            injected.extend(out if isinstance(out, list) else [out])
        injected = [f for f in injected if not f.affected_scope.get("benign")]
        if tier is not None:
            injected = [replace(f, difficulty_tier=tier) for f in injected]
        return con, injected
    return build


SCENARIOS: dict[str, Scenario] = {
    "healthy": Scenario(
        "healthy", "clean",
        "No fault injected: false-positive control scenario.",
        _make(),
    ),
    "bank_outage_icici": Scenario(
        "bank_outage_icici", "clean",
        "Issuer bank outage on ICICI.",
        _make(injector=BankOutageInjector(_base_window(), "ICICI")),
    ),
    "bank_outage_kotak": Scenario(
        "bank_outage_kotak", "clean",
        "Issuer bank outage on KOTAK (smallest issuer, lower volume).",
        _make(injector=BankOutageInjector(_base_window(), "KOTAK")),
    ),
    "network_degradation_visa": Scenario(
        "network_degradation_visa", "clean",
        "Card network degradation on visa across all issuers.",
        _make(injector=NetworkDegradationInjector(_base_window(), "visa")),
    ),
    "network_degradation_rupay": Scenario(
        "network_degradation_rupay", "clean",
        "Card network degradation on rupay (low volume network).",
        _make(injector=NetworkDegradationInjector(_base_window(), "rupay", failure_rate=0.50)),
    ),
    "high_ticket_rule_10k": Scenario(
        "high_ticket_rule_10k", "clean",
        "Risk rule starts declining amount > 10000 mid-window.",
        _make(injector=HighTicketRuleInjector(_base_window(), 10000)),
    ),
    "compound_outage_plus_rule": Scenario(
        "compound_outage_plus_rule", "compound",
        "ICICI outage overlapping with a >10000 rule trigger.",
        _make(injectors=[BankOutageInjector(_base_window(), "ICICI"),
                         HighTicketRuleInjector(_base_window(), 10000)]),
    ),
}


for _w2 in (
    Scenario(
        "retry_storm_gateway", "clean",
        "Diffuse retry storm: timeouts + duplicate attempts from mid-window.",
        _make(injector=RetryStormInjector(_base_window())),
    ),
    Scenario(
        "checkout_funnel_break", "clean",
        "Client-side checkout break: all payment methods fail with checkout_error.",
        _make(injector=CheckoutFunnelBreakInjector(_base_window())),
    ),
    Scenario(
        "settlement_delay_mch007", "clean",
        "One merchant's successes stall as pending; nothing actually fails.",
        _make(injector=SettlementDelayInjector(_base_window(), "mch_007")),
    ),
    Scenario(
        "red_herring_campaign_vs_outage", "red_herring",
        "Benign campaign volume spike in north co-occurs with a real HDFC outage.",
        _make(injectors=[MarketingSpikeInjector(_base_window(), "north"),
                         BankOutageInjector(_base_window(), "HDFC")],
              tier="red_herring"),
    ),
    Scenario(
        "benign_volume_spike", "red_herring",
        "False-positive control: campaign spike alone, healthy success rates.",
        _make(injector=MarketingSpikeInjector(_base_window(), "north")),
    ),
    Scenario(
        "noisy_bank_outage_hdfc", "noisy",
        "Partial HDFC outage (~62%), mixed decline codes.",
        _make(injector=BankOutageInjector(_base_window(), "HDFC", failure_rate=0.62,
                                          mixed_codes=True),
              tier="noisy"),
    ),
    Scenario(
        "noisy_network_amex", "noisy",
        "Amex degradation at the low end of the band and the lowest volume network.",
        _make(injector=NetworkDegradationInjector(_base_window(), "amex",
                                                  failure_rate=0.42),
              tier="noisy"),
    ),
):
    SCENARIOS[_w2.scenario_id] = _w2


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[scenario_id]


def save_ground_truth(path: str, scenario_id: str) -> dict:
    gt = get_scenario(scenario_id).ground_truth()
    with open(path, "w") as fh:
        json.dump(gt, fh, indent=2, default=str)
    return gt
