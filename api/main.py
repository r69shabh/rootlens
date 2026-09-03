"""RootLens diagnosis API.

POST /diagnose returns a verdict + evidence chain. The endpoint hides the LLM
provider (or rule baseline) behind a single entry point, so an internal user
or downstream service can call diagnosis without knowing the agent implementation.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_engine.generator import DEFAULT_WINDOW_START, WindowBounds, WindowConfig
from data_engine.scenarios import SCENARIOS, get_scenario
from diagnosis.agent import diagnose
from diagnosis.baseline_agent import rule_based_diagnose
from diagnosis.llm_client import get_client

app = FastAPI(title="RootLens", version="0.1.0")


class DiagnoseRequest(BaseModel):
    scenario_id: str
    # agent selector: "rule" (default, zero-LLM) or "openai:<model>" / "anthropic:<model>"
    agent: str = "rule"


class EvidenceEntry(BaseModel):
    call_id: str
    tool: str
    args: dict
    result: object
    row_count: int
    duration_ms: float
    ts: str


class DiagnoseResponse(BaseModel):
    status: str
    root_cause: str | None = None
    confidence: float | None = None
    evidence_call_ids: list[str] = Field(default_factory=list)
    disconfirmation: list[str] = Field(default_factory=list)
    impact: dict = Field(default_factory=dict)
    missing: str | None = None
    rounds_used: int = 0
    time_to_diagnosis_minutes: float | None = None
    evidence: list[EvidenceEntry] = Field(default_factory=list)


@app.get("/scenarios")
def list_scenarios():
    return [
        {"id": sid, "tier": sc.tier, "description": sc.description}
        for sid, sc in sorted(SCENARIOS.items())
    ]


def _bounds() -> WindowBounds:
    return WindowConfig(start=DEFAULT_WINDOW_START).bounds()


def _run(req: DiagnoseRequest) -> DiagnoseResponse:
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {req.scenario_id}")
    scenario = get_scenario(req.scenario_id)
    con, _ = scenario.build_dataset()
    try:
        b = _bounds()
        if req.agent == "rule":
            result = rule_based_diagnose(
                con,
                b.current_start,
                b.current_end,
                b.baseline_start,
                b.baseline_end,
                scenario_id=req.scenario_id,
            )
        else:
            provider, _, model = req.agent.partition(":")
            llm = get_client(provider, model or None)
            result = diagnose(
                con,
                b.current_start,
                b.current_end,
                b.baseline_start,
                b.baseline_end,
                llm=llm,
                scenario_id=req.scenario_id,
            )
    finally:
        con.close()
    return DiagnoseResponse(
        status=result.status,
        root_cause=result.root_cause,
        confidence=result.confidence,
        evidence_call_ids=result.evidence_call_ids,
        disconfirmation=result.disconfirmation,
        impact=result.impact,
        missing=result.missing,
        rounds_used=result.rounds_used,
        time_to_diagnosis_minutes=result.time_to_diagnosis_minutes,
        evidence=[EvidenceEntry(**vars(e)) for e in (result.store.entries if result.store else [])],
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
def post_diagnose(req: DiagnoseRequest):
    return _run(req)
