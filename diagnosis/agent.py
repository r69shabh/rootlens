"""Hand-rolled tool-calling loop (pipeline stages 2-4).

Week-2 hardening: verdict validation (evidence must exist in the audit trail,
disconfirmation mandatory), a midpoint disconfirmation nudge, and business-impact
estimation on every result. Max 8 rounds, temperature 0 via the LLM client.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from diagnosis.anomaly_scan import AnomalousSegment, estimate_onset, scan
from diagnosis.evidence import EvidenceStore
from diagnosis.impact import estimate_impact
from diagnosis.llm_client import LLMClient, parse_json_response
from diagnosis.priors import priors_context
from diagnosis.tools import DiagnosisTools, ToolError

MAX_TOOL_ROUNDS = 8
CONFIDENCE_THRESHOLD = 0.65

SYSTEM_PROMPT = """You are RootLens, a payments incident analyst. You NEVER invent numbers: every
number you state must come from a tool result in this conversation.

Tools available (JSON args):
- query_transactions(filters, group_by, metrics, start, end): filters/group_by may use
  issuer_bank, card_network, payment_method, amount_bucket, geo_region, merchant_id, status,
  failure_code; metrics: count, success_rate, failure_rate, avg_amount, p95_latency.
- timeseries(metric, granularity, start, end, filters): granularity hour|minute|day.
- compare_segments(dim_a, dim_b, start, end): check whether a signal concentrated in one
  dim_a slice is diffuse across dim_b slices.
- baseline_compare(metric, current_start, current_end, baseline_start, baseline_end, filters).

Respond with EXACTLY ONE JSON object per turn:
1. To call a tool: {"thought": "...", "tool": "<tool name>", "args": { ... }}
2. To give a verdict (only when confident > {conf}):
   {{"verdict": {{"root_cause": "<label like bank_outage:ICICI>", "confidence": <0..1>,
     "evidence": ["<call_id>", ...],
     "disconfirmation": ["what you checked that could have disproved this, and the result"],
     "impact": {{"transactions_affected": <n>, "note": "..."}}}}}}
3. If the hypothesis budget is exhausted without confidence:
   {{"inconclusive": {{"missing": "what data would resolve this"}}}}

Rules: for every hypothesis, actively try to DISCONFIRM it (e.g. if you suspect a bank outage,
check whether other banks on the same network are fine; if you suspect a network issue, check
whether the same bank's traffic on other networks is fine). A verdict is REJECTED if it cites
call_ids that do not exist in this conversation or has no disconfirmation checks. Do not
distinguish correlation from causation lightly: a volume spike in one region alongside a real
fault is a red herring unless it also shows degraded success rates AT the fault's location.
Never claim a number you did not observe. One tool call per turn.""".replace(
    "{conf}", str(CONFIDENCE_THRESHOLD)
)


@dataclass
class DiagnosisResult:
    status: str                      # "verdict" | "inconclusive" | "error"
    root_cause: str | None = None
    confidence: float | None = None
    evidence_call_ids: list[str] = field(default_factory=list)
    disconfirmation: list[str] = field(default_factory=list)
    impact: dict = field(default_factory=dict)
    missing: str | None = None
    rounds_used: int = 0
    time_to_diagnosis_minutes: float | None = None
    transcript: list[dict] = field(default_factory=list)
    store: EvidenceStore | None = None  # full audit trail travels with the result

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "evidence_call_ids": self.evidence_call_ids,
            "disconfirmation": self.disconfirmation,
            "impact": self.impact,
            "missing": self.missing,
            "rounds_used": self.rounds_used,
            "time_to_diagnosis_minutes": self.time_to_diagnosis_minutes,
        }


def build_initial_context(segments: list[AnomalousSegment], onset_hints: dict) -> str:
    lines = [
        "Success-rate drop detected in the current window. Ranked anomalous segments",
        "(deterministic scan; baseline vs current success rate, ranked by severity):",
    ]
    for s in segments[:8]:
        lines.append(
            f"- {s.dimension}={s.value}: baseline {s.baseline_rate:.1%}"
            f" -> current {s.current_rate:.1%}, "
            f"volume {s.current_volume} ({s.volume_share:.1%} of window), impact {s.impact:.3f}"
        )
    if not segments:
        lines.append("- (no segments crossed the anomaly thresholds)")
    lines.append("")
    lines.append(priors_context())
    if onset_hints:
        lines.append("Estimated onset times: " + json.dumps(onset_hints))
    return "\n".join(lines)


_DISCONFIRMATION_SUGGESTIONS = {
    "issuer_bank": ("compare_segments('issuer_bank','card_network', ...) to test whether the "
                    "failure is concentrated in one bank (outage) or spans a whole network "
                    "(network degradation); also check the failure_code distribution"),
    "card_network": ("compare_segments('card_network','issuer_bank', ...) — a true network "
                     "issue spans all issuers; a single-issuer concentration means a bank outage"),
    "amount_bucket": ("query_transactions(group_by=['failure_code']) — a risk rule shows one "
                      "dominant decline code tied to high amounts"),
    "merchant_id": ("query_transactions(filters={merchant}, group_by=['status']) — settlement "
                    "delays show a pending spike with failure_code NULL, not failed volume"),
    "payment_method": ("compare_segments('payment_method','failure_code', ...) — a checkout "
                       "funnel break hits all methods with one code; a gateway issue does not"),
}


def _disconfirmation_hint(segments: list[AnomalousSegment]) -> str | None:
    if not segments:
        return None
    base = _DISCONFIRMATION_SUGGESTIONS.get(segments[0].dimension)
    if not base:
        return None
    return ("Disconfirmation checkpoint: before concluding, actively try to DISPROVE the "
            f"leading hypothesis ({segments[0].dimension}={segments[0].value}). {base}")


def _verdict_problems(parsed: dict, store: EvidenceStore) -> list[str]:
    v = parsed["verdict"]
    problems: list[str] = []
    evidence = [str(c) for c in v.get("evidence", [])]
    if not evidence:
        problems.append("verdict cites no evidence; include the call_ids you relied on")
    for cid in evidence:
        if store.get(cid) is None:
            problems.append(f"evidence {cid!r} does not exist in the audit trail")
    if not v.get("disconfirmation"):
        problems.append("no disconfirmation checks; state what could have disproved the "
                        "hypothesis and what you found")
    try:
        conf = float(v.get("confidence", 0.0))
    except (TypeError, ValueError):
        problems.append("confidence is not a number")
    else:
        if not 0.0 <= conf <= 1.0:
            problems.append("confidence must be between 0 and 1")
        elif conf < CONFIDENCE_THRESHOLD:
            problems.append(
                f"confidence {conf} below threshold {CONFIDENCE_THRESHOLD}; keep investigating "
                "or return an explicit inconclusive"
            )
    return problems


def _extract_verdict(parsed: dict, result: DiagnosisResult, rounds: int) -> DiagnosisResult:
    v = parsed["verdict"]
    result.status = "verdict"
    result.root_cause = str(v.get("root_cause", "unknown"))
    result.confidence = float(v.get("confidence", 0.0))
    result.evidence_call_ids = [str(c) for c in v.get("evidence", [])]
    result.disconfirmation = [str(x) for x in v.get("disconfirmation", [])]
    result.impact = {"claimed_by_model": v.get("impact", {}) or {}}
    result.rounds_used = rounds
    return result


def diagnose(con, current_start, current_end, baseline_start, baseline_end,
             llm: LLMClient, scenario_id: str = "default",
             max_rounds: int = MAX_TOOL_ROUNDS) -> DiagnosisResult:
    """Full pipeline: deterministic scan -> LLM hypothesis/evidence loop -> verdict."""
    t0 = time.perf_counter()
    store = EvidenceStore(scenario_id=scenario_id)
    tools = DiagnosisTools(con, store)
    segments = scan(con, current_start, current_end, baseline_start, baseline_end)

    onset_hints = {}
    for s in segments[:3]:
        onset = estimate_onset(con, current_start, current_end, s.baseline_rate,
                               {s.dimension: s.value} if s.dimension != "amount_bucket" else {})
        if onset:
            onset_hints[f"{s.dimension}={s.value}"] = onset

    result = DiagnosisResult(status="error")
    context = build_initial_context(segments, onset_hints)
    messages: list[dict] = [{"role": "user", "content": context}]
    transcript: list[dict] = [{"role": "context", "content": context}]
    hint_shown = False
    rejected_problems: list[str] = []
    hint = _disconfirmation_hint(segments)

    for round_no in range(1, max_rounds + 1):
        reply = llm.chat(SYSTEM_PROMPT, messages)
        transcript.append({"role": "assistant", "content": reply})
        try:
            parsed = parse_json_response(reply)
        except ValueError as exc:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": f"Invalid JSON: {exc}. Reply with one JSON object."})
            continue

        if "verdict" in parsed:
            problems = _verdict_problems(parsed, store)
            if problems and round_no < max_rounds:
                transcript.append({"role": "verdict_rejected", "problems": problems})
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "VERDICT_REJECTED: "
                                 + "; ".join(problems) + ". Fix these and respond again."})
                continue
            if problems:
                rejected_problems = problems
                break
            result = _extract_verdict(parsed, result, round_no)
            break

        if "inconclusive" in parsed:
            result = DiagnosisResult(
                status="inconclusive", missing=str(parsed["inconclusive"].get("missing", "")),
                rounds_used=round_no,
            )
            break

        if "tool" not in parsed:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": 'Reply with a tool call, {"verdict": ...} or '
                                        '{"inconclusive": ...}.'})
            continue

        tool_name, args = parsed["tool"], parsed.get("args", {})
        try:
            tool_result = tools.dispatch(tool_name, args)
            observation = json.dumps(tool_result, default=str)
        except (ToolError, TypeError) as exc:
            observation = f"TOOL_ERROR: {exc}"
            store.log(tool_name, args, observation, 0.0)
        except Exception as exc:  # noqa: BLE001 - surface tool crash, keep loop alive
            observation = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
            store.log(tool_name, args, observation, 0.0)

        transcript.append({"role": "tool", "tool": tool_name,
                           "args": args, "result": observation})
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user",
                         "content": f"Tool result for {tool_name}:\n{observation}"})

        if not hint_shown and hint and round_no >= max(1, max_rounds // 2):
            hint_shown = True
            transcript.append({"role": "nudge", "content": hint})
            messages.append({"role": "user", "content": hint})
    else:
        result = DiagnosisResult(
            status="inconclusive",
            missing=(f"hypothesis budget of {max_rounds} tool rounds exhausted "
                     f"without a confident verdict"),
            rounds_used=max_rounds,
        )

    if rejected_problems:
        result = DiagnosisResult(
            status="inconclusive",
            missing="final verdict rejected: " + "; ".join(rejected_problems),
            rounds_used=max_rounds,
        )

    elapsed_minutes = (time.perf_counter() - t0) / 60
    result.time_to_diagnosis_minutes = round(elapsed_minutes, 2)
    result.impact = {"estimated": estimate_impact(
        con, current_start, current_end, elapsed_minutes), **result.impact}
    result.transcript = transcript
    result.store = store
    return result
