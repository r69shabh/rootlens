#!/usr/bin/env python3
"""Week-3 eval runner: all scenarios x agents -> tiered accuracy + leaderboard.

Rule baseline (no LLM needed):
    python scripts/run_eval.py --agent rule
A live provider (needs API key; records nothing by default):
    python scripts/run_eval.py --agent openai:gpt-4o-mini
Only some scenarios:
    python scripts/run_eval.py --agent rule --scenarios clean,compound
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import UTC, datetime  # noqa: E402

from data_engine.generator import WindowConfig  # noqa: E402
from data_engine.scenarios import SCENARIOS  # noqa: E402
from diagnosis.agent import diagnose  # noqa: E402
from diagnosis.baseline_agent import rule_based_diagnose  # noqa: E402
from diagnosis.llm_client import get_client  # noqa: E402
from eval.harness import score_batch, score_result  # noqa: E402
from eval.leaderboard import Leaderboard, RunRecord  # noqa: E402


def ground_truth_for(scenario) -> dict:
    con, faults = scenario.build_dataset()
    con.close()
    return {"scenario_id": scenario.scenario_id,
            "difficulty_tier": scenario.tier,
            "expected_labels": [f.label for f in faults],
            "expected_fault_types": sorted({f.fault_type for f in faults})}


def run(agent: str, scenario_ids: list[str] | None = None,
        out_dir: str = "eval/results") -> dict:
    ids = scenario_ids or sorted(SCENARIOS)
    board = Leaderboard()
    details = []
    scored: list[tuple] = []  # (DiagnosisResult, ground_truth)

    for sid in ids:
        scenario = SCENARIOS[sid]
        gt = ground_truth_for(scenario)
        con, _ = scenario.build_dataset()
        wc = WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))
        # order matches (current_start, current_end, baseline_start, baseline_end)
        bounds = (wc.current_window_start, wc.current_window_end,
                  wc.start, wc.current_window_start)

        if agent == "rule":
            result = rule_based_diagnose(
                con, *bounds, scenario_id=sid)
            usage, calls = [], 0
        else:
            provider = agent.split(":")[0]
            llm = get_client(provider, model=agent.split(":", 1)[1] if ":" in agent else None)
            result = diagnose(con, *bounds, llm=llm, scenario_id=sid)
            usage, calls = llm.usage, len(llm.usage)

        score = score_result(result, gt)
        board.add(RunRecord(
            scenario_id=sid, tier=gt["difficulty_tier"], agent=agent,
            status=result.status, correct=score["correct"],
            partial=score.get("partial", False),
            inconclusive=score["inconclusive"],
            false_positive=score.get("false_positive", False),
            latency_minutes=result.time_to_diagnosis_minutes or 0.0,
            llm_calls=calls,
            input_tokens=sum(u["input_tokens"] for u in usage),
            output_tokens=sum(u["output_tokens"] for u in usage),
            tokens_estimated=any(u.get("estimated") for u in usage),
        ))
        details.append({"scenario_id": sid, "tier": gt["difficulty_tier"],
                        "score": score, "result": result.to_json()})
        scored.append((result, gt))
        flag = "OK " if score["correct"] else ("INC" if result.status == "inconclusive" else "MISS")
        print(f"  [{flag}] {sid:36s} -> {result.root_cause or result.status}")
        con.close()

    tier_reports = score_batch(scored)
    print()
    for tier in sorted(tier_reports):
        print("  " + tier_reports[tier].summary())
    print()
    for standing in board.standings():
        print("  " + standing.summary())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{agent.replace(':', '_')}_results.json").write_text(json.dumps(
        {"agent": agent, "details": details,
         "standings": [vars(s) for s in board.standings()]},
        indent=2, default=str))
    (out / f"{agent.replace(':', '_')}_leaderboard.md").write_text(board.to_markdown())
    print(f"\nresults written to {out}/")
    return {"tier_reports": tier_reports, "board": board}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="rule",
                    help="'rule' or 'openai' / 'anthropic' (optionally 'openai:gpt-4o-mini')")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated scenario ids or tier names")
    ns = ap.parse_args()

    ids = None
    if ns.scenarios:
        wanted = set(ns.scenarios.split(","))
        ids = [sid for sid in sorted(SCENARIOS)
               if sid in wanted or SCENARIOS[sid].tier in wanted]
    results = run(ns.agent, ids)
    # exit non-zero if clean-tier accuracy is below target (architecture: >85%)
    clean = results["tier_reports"].get("clean")
    return 0 if clean is None or clean.accuracy >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
