"""Fault injectors. Each applies a deterministic signature and returns ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_engine.generator import CARD_NETWORKS, COLUMNS, ISSUER_BANKS


@dataclass(frozen=True)
class InjectedFault:
    fault_type: str
    label: str
    start_ts: Any
    end_ts: Any
    affected_scope: dict
    difficulty_tier: str


class FaultInjector:
    """Base class: injects into the generator's current window, post-generation."""

    fault_type = "base"
    tier = "clean"

    def __init__(self, window) -> None:
        self.window = window

    def start_ts(self):
        return self.window.current_window_start

    def end_ts(self):
        return self.window.current_window_end

    def inject(self, con):  # pragma: no cover - interface
        raise NotImplementedError


class BankOutageInjector(FaultInjector):
    """Elevated failure rate for one issuer_bank in the window.

    Default ~90% (clean tier). `failure_rate` + `mixed_codes` give the noisy tier
    a partial, ambiguous version of the same fault.
    """

    fault_type = "bank_outage"
    tier = "clean"

    def __init__(self, window, bank: str, failure_rate: float = 0.90,
                 mixed_codes: bool = False) -> None:
        super().__init__(window)
        if bank not in ISSUER_BANKS:
            raise ValueError(f"unknown bank {bank!r}; expected one of {ISSUER_BANKS}")
        if not 0.05 <= failure_rate <= 0.95:
            raise ValueError("failure_rate must be in [0.05, 0.95]")
        self.bank = bank
        self.failure_rate = failure_rate
        self.mixed_codes = mixed_codes

    def inject(self, con) -> InjectedFault:
        con.execute(
            """
            UPDATE transactions
            SET status = 'failed',
                failure_code = CASE WHEN ? AND hash(txn_id) % 10 < 3
                                    THEN 'issuer_declined'
                                    ELSE 'issuer_unavailable' END,
                gateway_latency_ms = gateway_latency_ms + 2000
            WHERE issuer_bank = ? AND ts >= ? AND ts < ?
              AND hash(txn_id) % 100 < ?
            """,
            [self.mixed_codes, self.bank, self.start_ts(), self.end_ts(),
             int(self.failure_rate * 100)],
        )
        if not self.mixed_codes:
            # keep a slice succeeding so the signal is realistic, not absolute
            con.execute(
                """
                UPDATE transactions
                SET status = 'success', failure_code = NULL
                WHERE issuer_bank = ? AND ts >= ? AND ts < ?
                  AND hash(txn_id) % 10 < 1
                """,
                [self.bank, self.start_ts(), self.end_ts()],
            )
        return InjectedFault(
            fault_type=self.fault_type,
            label=f"bank_outage:{self.bank}",
            start_ts=self.start_ts(),
            end_ts=self.end_ts(),
            affected_scope={"issuer_bank": self.bank, "failure_rate": self.failure_rate},
            difficulty_tier=self.tier,
        )


class NetworkDegradationInjector(FaultInjector):
    """40-55% failure rate across ALL issuers on one card_network."""

    fault_type = "network_degradation"
    tier = "clean"

    def __init__(self, window, network: str, failure_rate: float = 0.48) -> None:
        super().__init__(window)
        if network not in CARD_NETWORKS:
            raise ValueError(f"unknown network {network!r}; expected one of {CARD_NETWORKS}")
        if not 0.40 <= failure_rate <= 0.55:
            raise ValueError("architecture specifies 40-55% failure rate for this fault")
        self.network = network
        self.failure_rate = failure_rate

    def inject(self, con) -> InjectedFault:
        # deterministic per-txn coin via hash, so re-runs on the same DB are stable
        con.execute(
            """
            UPDATE transactions
            SET status = 'failed', failure_code = 'network_declined',
                gateway_latency_ms = gateway_latency_ms + 1200
            WHERE payment_method = 'card' AND card_network = ?
              AND ts >= ? AND ts < ?
              AND hash(txn_id) % 100 < ?
            """,
            [self.network, self.start_ts(), self.end_ts(), int(self.failure_rate * 100)],
        )
        return InjectedFault(
            fault_type=self.fault_type,
            label=f"network_degradation:{self.network}",
            start_ts=self.start_ts(),
            end_ts=self.end_ts(),
            affected_scope={"card_network": self.network},
            difficulty_tier=self.tier,
        )


class HighTicketRuleInjector(FaultInjector):
    """60-80% failure rate for amount > threshold after a mid-window rule change."""

    fault_type = "high_ticket_rule"
    tier = "clean"

    def __init__(self, window, threshold: float, failure_rate: float = 0.70) -> None:
        super().__init__(window)
        if not 0.60 <= failure_rate <= 0.80:
            raise ValueError("architecture specifies 60-80% failure rate for this fault")
        self.threshold = threshold
        self.failure_rate = failure_rate
        span = window.current_window_end - window.current_window_start
        self.rule_start = window.current_window_start + span / 2

    def start_ts(self):
        return self.rule_start

    def inject(self, con) -> InjectedFault:
        con.execute(
            """
            UPDATE transactions
            SET status = 'failed', failure_code = 'rule_declined',
                gateway_latency_ms = gateway_latency_ms + 100
            WHERE amount > ? AND ts >= ? AND ts < ?
              AND hash(txn_id) % 100 < ?
            """,
            [self.threshold, self.rule_start, self.end_ts(), int(self.failure_rate * 100)],
        )
        return InjectedFault(
            fault_type=self.fault_type,
            label=f"rule_trigger:{int(self.threshold)}",
            start_ts=self.rule_start,
            end_ts=self.end_ts(),
            affected_scope={"amount_gt": self.threshold, "started_at": self.rule_start},
            difficulty_tier=self.tier,
        )


class RetryStormInjector(FaultInjector):
    """Diffuse degradation from mid-window: elevated timeouts + a volume spike of
    duplicate retry attempts. Signature: latency up, `gateway_timeout` codes,
    transaction count inflated by retry duplicates."""

    fault_type = "retry_storm"
    tier = "clean"

    def __init__(self, window, failure_rate: float = 0.35,
                 retry_fraction: float = 0.40) -> None:
        super().__init__(window)
        self.failure_rate = failure_rate
        self.retry_fraction = retry_fraction
        span = window.current_window_end - window.current_window_start
        self.storm_start = window.current_window_start + span / 2

    def start_ts(self):
        return self.storm_start

    def inject(self, con) -> InjectedFault:
        con.execute(
            """
            UPDATE transactions
            SET status = 'failed', failure_code = 'gateway_timeout',
                gateway_latency_ms = gateway_latency_ms + 1500
            WHERE ts >= ? AND ts < ? AND hash(txn_id) % 100 < ?
            """,
            [self.storm_start, self.end_ts(), int(self.failure_rate * 100)],
        )
        con.execute(
            f"""
            INSERT INTO transactions ({', '.join(COLUMNS)})
            SELECT txn_id || '_retry', ts, amount, currency, payment_method,
                   card_network, issuer_bank, status, failure_code,
                   gateway_latency_ms + 1500, merchant_id, geo_region
            FROM transactions
            WHERE ts >= ? AND ts < ? AND hash(txn_id) % 100 < ?
            """,
            [self.storm_start, self.end_ts(), int(self.retry_fraction * 100)],
        )
        return InjectedFault(
            fault_type=self.fault_type,
            label="retry_storm:gateway",
            start_ts=self.storm_start,
            end_ts=self.end_ts(),
            affected_scope={"scope": "all_traffic", "started_at": self.storm_start,
                            "retry_fraction": self.retry_fraction},
            difficulty_tier=self.tier,
        )


class CheckoutFunnelBreakInjector(FaultInjector):
    """Client-side checkout break: ~50% of ALL payment methods fail with
    `checkout_error` from mid-window. Latency unchanged (not a gateway problem)."""

    fault_type = "checkout_funnel_break"
    tier = "clean"

    def __init__(self, window, failure_rate: float = 0.50) -> None:
        super().__init__(window)
        self.failure_rate = failure_rate
        span = window.current_window_end - window.current_window_start
        self.break_start = window.current_window_start + span / 2

    def start_ts(self):
        return self.break_start

    def inject(self, con) -> InjectedFault:
        con.execute(
            """
            UPDATE transactions
            SET status = 'failed', failure_code = 'checkout_error'
            WHERE ts >= ? AND ts < ? AND hash(txn_id) % 100 < ?
            """,
            [self.break_start, self.end_ts(), int(self.failure_rate * 100)],
        )
        return InjectedFault(
            fault_type=self.fault_type,
            label="checkout_funnel_break:checkout",
            start_ts=self.break_start,
            end_ts=self.end_ts(),
            affected_scope={"scope": "all_traffic", "started_at": self.break_start},
            difficulty_tier=self.tier,
        )


class SettlementDelayInjector(FaultInjector):
    """One merchant's successful transactions stall as 'pending' from mid-window.
    Distinctive signature: failure_code stays NULL, nothing 'failed' — the drop is
    entirely pending volume."""

    fault_type = "settlement_delay"
    tier = "clean"

    def __init__(self, window, merchant_id: str = "mch_007",
                 pending_rate: float = 0.80) -> None:
        super().__init__(window)
        self.merchant_id = merchant_id
        self.pending_rate = pending_rate
        span = window.current_window_end - window.current_window_start
        self.delay_start = window.current_window_start + span / 2

    def start_ts(self):
        return self.delay_start

    def inject(self, con) -> InjectedFault:
        con.execute(
            """
            UPDATE transactions
            SET status = 'pending'
            WHERE merchant_id = ? AND ts >= ? AND ts < ?
              AND status = 'success' AND hash(txn_id) % 100 < ?
            """,
            [self.merchant_id, self.delay_start, self.end_ts(),
             int(self.pending_rate * 100)],
        )
        return InjectedFault(
            fault_type=self.fault_type,
            label=f"settlement_delay:{self.merchant_id}",
            start_ts=self.delay_start,
            end_ts=self.end_ts(),
            affected_scope={"merchant_id": self.merchant_id,
                            "pending_rate": self.pending_rate},
            difficulty_tier=self.tier,
        )


class MarketingSpikeInjector(FaultInjector):
    """BENIGN correlated anomaly (red herring): a campaign triples one region's
    volume with no change in success rate. Never recorded as ground truth."""

    fault_type = "benign_volume_spike"
    tier = "red_herring"

    def __init__(self, window, geo_region: str = "north",
                 extra_fraction: float = 1.0) -> None:
        super().__init__(window)
        self.geo_region = geo_region
        self.extra_fraction = extra_fraction

    def inject(self, con) -> InjectedFault:
        con.execute(
            f"""
            INSERT INTO transactions ({', '.join(COLUMNS)})
            SELECT txn_id || '_camp', ts, amount, currency, payment_method,
                   card_network, issuer_bank, status, failure_code,
                   gateway_latency_ms, merchant_id, geo_region
            FROM transactions
            WHERE geo_region = ? AND ts >= ? AND ts < ?
              AND hash(txn_id) % 100 < ?
            """,
            [self.geo_region, self.start_ts(), self.end_ts(),
             int(self.extra_fraction * 100)],
        )
        return InjectedFault(
            fault_type=self.fault_type,
            label=f"benign_volume_spike:{self.geo_region}",
            start_ts=self.start_ts(),
            end_ts=self.end_ts(),
            affected_scope={"geo_region": self.geo_region, "benign": True},
            difficulty_tier=self.tier,
        )


class CompoundFaultInjector(FaultInjector):
    """Bank outage + high-ticket rule overlapping in time (compound tier)."""

    fault_type = "compound"
    tier = "compound"

    def __init__(self, window, bank: str, threshold: float) -> None:
        super().__init__(window)
        self.bank_inj = BankOutageInjector(window, bank)
        self.rule_inj = HighTicketRuleInjector(window, threshold)

    def inject(self, con) -> list[InjectedFault]:
        return [self.bank_inj.inject(con), self.rule_inj.inject(con)]
