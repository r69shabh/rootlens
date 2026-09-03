"""RootLens evidence-chain UI (Streamlit).

Run: uv run --extra ui streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import UTC, datetime

import streamlit as st

from data_engine.generator import WindowConfig
from data_engine.scenarios import SCENARIOS, get_scenario
from diagnosis.agent import diagnose
from diagnosis.llm_client import get_client
from diagnosis.replay import ReplayCache, ReplayLLMClient
from eval.harness import score_result
from eval.report import to_markdown

st.set_page_config(page_title="RootLens", page_icon="🔍", layout="wide")
st.title("🔍 RootLens — payment failure root-cause diagnosis")
st.caption("Every number traceable to a tool call. No LLM-invented evidence.")


def _window():
    wc = WindowConfig(start=datetime(2026, 8, 24, tzinfo=UTC))
    return wc.bounds()


with st.sidebar:
    st.header("Run")
    scenario_id = st.selectbox(
        "Scenario", sorted(SCENARIOS),
        format_func=lambda sid: f"{sid} ({SCENARIOS[sid].tier})",
    )
    mode = st.radio("LLM mode", ["Replay cache", "Live provider"])
    provider = None
    cache_path = "data/replay_cache.json"
    if mode == "Live provider":
        provider = st.selectbox("Provider", ["openai", "anthropic"])
        st.caption("Key read from OPENAI_API_KEY / ANTHROPIC_API_KEY")
    else:
        cache_path = st.text_input("Replay cache file", cache_path)
    if st.button("Diagnose", type="primary", use_container_width=True):
        with st.spinner("Running diagnosis loop..."):
            scenario = get_scenario(scenario_id)
            con, faults = scenario.build_dataset()
            gt = {"scenario_id": scenario.scenario_id,
                  "difficulty_tier": scenario.tier,
                  "expected_labels": [f.label for f in faults],
                  "expected_fault_types": sorted({f.fault_type for f in faults})}
            bounds = _window()
            b0, b1, c0, c1 = (bounds.baseline_start, bounds.baseline_end,
                              bounds.current_start, bounds.current_end)
            if mode == "Replay cache":
                llm = ReplayLLMClient(ReplayCache(cache_path))
            else:
                llm = get_client(provider)
            result = diagnose(con, c0, c1, b0, b1, llm=llm,
                              scenario_id=scenario_id)
            st.session_state["result"] = result
            st.session_state["gt"] = gt
            con.close()

if "result" not in st.session_state:
    st.info("Pick a scenario and hit **Diagnose**. Replay mode needs a cache file "
            "produced via `scripts/run_scenario.py --record`.")
    st.stop()

result = st.session_state["result"]
gt = st.session_state["gt"]
score = score_result(result, gt)

# --- verdict card -----------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Status", result.status)
col2.metric("Root cause", result.root_cause or "—")
col3.metric("Confidence", f"{result.confidence:.0%}" if result.confidence else "—")

if gt["expected_labels"]:
    ok = "✅ correct" if score["correct"] else (
        "🟡 partial" if score.get("partial") else "❌ incorrect")
    st.caption(f"Eval (tier `{gt['difficulty_tier']}`): {ok} — "
               f"expected {gt['expected_labels']}")

if result.status == "inconclusive":
    st.warning(f"**Inconclusive** — {result.missing}")

# --- business impact --------------------------------------------------------
est = result.impact.get("estimated", {})
if est:
    st.subheader("Business impact")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("GMV at risk (upper bound)", f"₹{est.get('gmv_at_risk_inr', 0):,.0f}")
    i2.metric("Txns affected", est.get("non_success_txns_in_window", 0))
    i3.metric("Time to diagnosis", f"{est.get('time_to_diagnosis_minutes', 0)} min")
    i4.metric("Hours saved vs manual", est.get("hours_saved_vs_manual", 0))
    st.caption(est.get("note", ""))

# --- disconfirmation --------------------------------------------------------
if result.disconfirmation:
    st.subheader("Disconfirmation checks (what could have disproved it)")
    for d in result.disconfirmation:
        st.markdown(f"- {d}")

# --- evidence chain ----------------------------------------------------------
store = result.store
st.subheader(f"Evidence chain ({len(result.evidence_call_ids)} cited of "
             f"{len(store.entries)} logged calls)")
for entry in store.entries:
    cited = "✅ cited" if entry.call_id in result.evidence_call_ids else "logged"
    with st.expander(f"{entry.call_id} · {entry.tool} · {entry.row_count} rows · {cited}"):
        st.json({"args": entry.args, "result": entry.result}, expanded=True)

# --- transcript --------------------------------------------------------------
with st.expander("Full transcript (audit trail)"):
    for t in result.transcript:
        st.json(t, expanded=False)

# --- export -------------------------------------------------------------------
st.download_button(
    "Download diagnosis (JSON)",
    data=json.dumps({"result": result.to_json(), "score": score,
                     "evidence": store.to_json()}, indent=2, default=str),
    file_name=f"{gt['scenario_id']}_diagnosis.json",
)
st.download_button(
    "Download report (Markdown)",
    data=to_markdown(result, store, gt),
    file_name=f"{gt['scenario_id']}_report.md",
)
