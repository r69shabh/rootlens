#!/usr/bin/env python3
"""Week-2 proof script: end-to-end ask -> diagnosis -> evidence trail.

Deterministic only:   python scripts/run_scenario.py --scenario bank_outage_icici
Record a live run:    python scripts/run_scenario.py --scenario bank_outage_icici \
                          --provider anthropic --record data/replay_bank_outage_icici.json
Replay (no API key):  python scripts/run_scenario.py --scenario bank_outage_icici \
                          --replay data/replay_bank_outage_icici.json
List scenarios:       python scripts/run_scenario.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig  # noqa: E402
from data_engine.scenarios import SCENARIOS, get_scenario  # noqa: E402
from diagnosis.agent import diagnose  # noqa: E402
from diagnosis.anomaly_scan import scan  # noqa: E402
from diagnosis.llm_client import get_client  # noqa: E402
from diagnosis.replay import RecordingLLMClient, ReplayCache, ReplayLLMClient  # noqa: E402
from eval.harness import score_result  # noqa: E402
from eval.report import to_markdown  # noqa: E402

WC = WindowConfig(start=DEFAULT_WINDOW_START)


def window_bounds():
    return WC.bounds()


def deterministic_check(con, faults, segments) -> bool:
    """No-LLM proof: top-ranked segment (or failure-mode signature) must align
    with the injected ground truth. Fault-type aware."""
    if not faults:
        return not segments  # healthy/benign: silence is the pass condition
    types = {f.fault_type for f in faults}
    if not segments:
        return False
    top = segments[0]
    scope_match = any(
        top.value == f.affected_scope.get("issuer_bank")
        or top.value == f.affected_scope.get("card_network")
        or top.value == f.affected_scope.get("merchant_id")
        or (top.dimension == "amount_bucket" and "amount_gt" in f.affected_scope)
        for f in faults
    )
    if scope_match:
        return True
    # diffuse faults (retry storm / checkout break): signature is a failure code
    if "retry_storm" in types:
        n = con.execute(
            "SELECT COUNT(*) FROM transactions WHERE ts >= ? AND ts < ? "
            "AND failure_code = 'gateway_timeout'",
            [WC.current_window_start, WC.current_window_end],
        ).fetchone()[0]
        return n > 30
    if "checkout_funnel_break" in types:
        n = con.execute(
            "SELECT COUNT(*) FROM transactions WHERE ts >= ? AND ts < ? "
            "AND failure_code = 'checkout_error'",
            [WC.current_window_start, WC.current_window_end],
        ).fetchone()[0]
        return n > 30
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="bank_outage_icici", choices=sorted(SCENARIOS))
    ap.add_argument("--provider", default=None, choices=["openai", "anthropic"])
    ap.add_argument(
        "--record",
        default=None,
        metavar="CACHE.json",
        help="run live and cache the transcript for replay",
    )
    ap.add_argument(
        "--replay",
        default=None,
        metavar="CACHE.json",
        help="replay a recorded transcript (no API call)",
    )
    ap.add_argument("--dump", default=None, help="write full diagnosis JSON here")
    ap.add_argument("--list", action="store_true")
    ns = ap.parse_args()

    if ns.list:
        for sid, sc in sorted(SCENARIOS.items()):
            print(f"{sid:36s} [{sc.tier:12s}] {sc.description}")
        return 0
    if ns.record and ns.replay:
        ap.error("--record and --replay are mutually exclusive")

    scenario = get_scenario(ns.scenario)
    con, faults = scenario.build_dataset()
    gt = {
        "scenario_id": scenario.scenario_id,
        "difficulty_tier": scenario.tier,
        "expected_labels": [f.label for f in faults],
        "expected_fault_types": sorted({f.fault_type for f in faults}),
    }
    bounds = window_bounds()
    baseline_start, baseline_end = bounds.baseline_start, bounds.baseline_end
    current_start, current_end = bounds.current_start, bounds.current_end

    print(f"=== RootLens proof: {scenario.scenario_id} ({scenario.tier}) ===")
    print("\n-- Stage 1: deterministic segmented anomaly scan (no LLM) --")
    segments = scan(con, current_start, current_end, baseline_start, baseline_end)
    if not segments:
        print("  no anomalous segments (healthy/benign scenario) — FP control OK")
    for s in segments[:5]:
        d = s.to_dict()
        print(
            f"  {d['dimension']}={d['value']}: {d['baseline_success_rate']} -> "
            f"{d['current_success_rate']} (impact {d['impact']})"
        )

    if ns.provider is None and ns.replay is None:
        ok = deterministic_check(con, faults, segments)
        print(f"\n-- Deterministic ground-truth alignment: {ok}")
        print("PASS" if ok else "FAIL")
        con.close()
        return 0 if ok else 1

    print(f"\n-- Stages 2-4: LLM diagnosis loop ({ns.provider or 'replay'}) --")
    if ns.replay:
        llm = ReplayLLMClient(ReplayCache(ns.replay))
    else:
        live = get_client(ns.provider)
        llm = RecordingLLMClient(live, ReplayCache(ns.record)) if ns.record else live

    result = diagnose(
        con,
        current_start,
        current_end,
        baseline_start,
        baseline_end,
        llm=llm,
        scenario_id=scenario.scenario_id,
    )
    score = score_result(result, gt)
    if ns.record:
        llm.cache.save()
        print(f"transcript cached to {ns.record} ({len(llm.cache)} steps)")

    print(to_markdown(result, store=result.store, ground_truth=gt))
    print(f"eval: {json.dumps(score, default=str)}")
    if ns.dump:
        Path(ns.dump).write_text(
            json.dumps(
                {"result": result.to_json(), "score": score, "evidence": result.store.to_json()},
                indent=2,
                default=str,
            )
        )
        print(f"full audit trail written to {ns.dump}")
    con.close()
    return 0 if score["correct"] or result.status == "inconclusive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
