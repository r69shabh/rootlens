# RootLens

RootLens is an agent that diagnoses the root cause of payment failures, grounded entirely in deterministic evidence rather than LLM guesswork. Every claim in a diagnosis report is traceable to a specific tool call and its result (a full audit trail), and diagnosis accuracy is measured against a synthetic dataset with known, injected faults — not just demoed on cherry-picked examples.

See [`docs/architecture.md`](docs/architecture.md) for the full design and 3-week build plan.

## Status — Weeks 1–3 complete

- [x] Synthetic data engine: seeded generator → DuckDB (`transactions`), fault events (`fault_events`), ground-truth JSON
- [x] Fault injectors: bank outage, network degradation, high-ticket rule, compound — with scope-isolation and rate-band guarantees tested
- [x] Deterministic segmented anomaly scan (no LLM): all dimensions, ranked anomalous slices, onset estimation, false-positive control on healthy windows
- [x] Historical fault-rate priors as structured LLM context
- [x] Read-only whitelisted SQL tool set (`query_transactions`, `timeseries`, `compare_segments`, `baseline_compare`), fully parameterized
- [x] Evidence store / audit trail: every tool call logged, citable by `call_id`
- [x] `LLMClient` adapter: OpenAI + Anthropic (lazy extras), scripted client for tests/replay, temp-0 protocol in the prompt (model-agnostic)
- [x] Hand-rolled tool-calling loop: max 8 rounds, disconfirmation required, explicit inconclusive verdicts
- [x] Eval harness: per-tier scoring, honest inconclusive/false-positive reporting

Week-1 exit criterion met on all 7 original scenarios.

**Week 2 additions:**

- [x] New fault types: retry storm (timeouts + duplicate-attempt volume spike), checkout funnel break (one code across all methods), settlement delay (pending spike, nothing actually fails)
- [x] Red-herring tier: benign campaign volume spike co-occurring with a real fault — the spike is never ground truth, and the scan still ranks the real fault first
- [x] Noisy tier: partial bank outage (~62%, mixed decline codes) and low-volume network degradation — both still detected
- [x] Statistically-gated anomaly scan: two-proportion z-test with a conservative threshold (~40 slices tested per scan) — zero false positives on healthy AND benign-spike windows
- [x] Verdict validation: cited call_ids must exist in the audit trail, disconfirmation checks mandatory, sub-threshold confidence rejected with a pushback round
- [x] Midpoint disconfirmation nudges, tailored to the leading segment's dimension
- [x] Business impact translation: GMV at risk (explicit upper bound, all non-success txns in window), hours saved vs a documented 45-min manual baseline — deterministic to the cent
- [x] Replay mode: per-prompt transcript cache; record once, replay forever; cache misses are loud errors, never silent live calls
- [x] Markdown report export + Streamlit evidence-chain UI (verdict card, impact metrics, expandable per-call evidence, full transcript, JSON/MD export)

**Week 3 additions:**

- [x] Rule-based baseline agent (`diagnosis/baseline_agent.py`): zero-LLM diagnosis with real evidence-chain tool calls — the leaderboard floor
- [x] Full eval runner (`scripts/run_eval.py`): all scenarios × agents, per-tier accuracy, honest inconclusive/miss reporting, exits non-zero below the 85% clean-tier target
- [x] Model leaderboard (`eval/leaderboard.py`): accuracy × latency × cost per agent, per-tier breakdown never blended; token usage captured from provider responses (scripted client flagged as estimated)
- [x] Healthy/benign controls scored correctly: staying silent is a pass, raising a verdict is a false positive
- [x] 5-minute pitch script with demo-day checklist (`docs/demo_script.md`)

**Measured results (rule baseline, current build):**

| Tier | Accuracy | Notes |
|---|---|---|
| clean | 9/9 = 100% | incl. healthy + benign-spike FP controls (0 false positives) |
| compound | 1/1 = 100% | outage + rule trigger both identified |
| noisy | 2/2 = 100% | partial outage, low-volume network |
| red_herring | 2/2 = 100% | blames the real fault, not the correlated spike |

Week-3 exit: eval harness + leaderboard shipped, pitch script ready for recording.

## Quickstart

All commands assume you are in the repo root.

```bash
uv sync                        # base install (tests, scripts, deterministic pipeline)
uv run pytest                  # 96 tests, ~25s
```

### 1. Deterministic demo (no API key needed)

```bash
uv run python scripts/run_scenario.py --list                 # all 14 scenarios
uv run python scripts/run_scenario.py --scenario bank_outage_icici   # scan proof, PASS/FAIL
uv run python scripts/run_eval.py --agent rule               # full 14-scenario eval + leaderboard
```

### 2. Live LLM run + replay (one API key, used once)

Providers: `gemini` (default model `gemini-3.6-flash`, override with `GEMINI_MODEL`),
`openai`, `anthropic`. Keys via env: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`),
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Install provider SDKs first:

```bash
uv sync --extra llm
export GEMINI_API_KEY="..."
mkdir -p data
# record: runs the real agent loop, caches every transcript step
uv run python scripts/run_scenario.py --scenario bank_outage_icici \
    --provider gemini --record data/replay_bank_outage_icici.json --dump /tmp/live.json
# replay: byte-identical diagnosis, zero API calls, no key needed
uv run python scripts/run_scenario.py --scenario bank_outage_icici \
    --replay data/replay_bank_outage_icici.json
```

Notes: free-tier quotas are rate-limited (Gemini: 5 req/min) — the client backs
off automatically, so recording takes a couple of minutes. `data/` is
gitignored; the cache file lives only on your disk, back it up before a demo.
A cache miss is a loud error, never a silent live call.

### 3. Streamlit evidence-chain UI

```bash
uv run --extra ui streamlit run frontend/app.py
```

Sidebar: pick scenario → **Replay cache** → path `data/replay_bank_outage_icici.json`
→ **Diagnose**. Verdict card, business impact, expandable per-call evidence, full
transcript, JSON/Markdown export. **Live provider** mode needs the matching env key.
If the cache path is wrong you get an explicit "file not found" error, not a crash.

### 4. FastAPI service

```bash
uv sync --extra api
uv run uvicorn api.main:app --reload
# GET  /scenarios            # list all 14 scenarios
# POST /diagnose  {"scenario_id": "bank_outage_icici", "agent": "rule"}
#   agent: "rule" (zero-LLM) or "gemini" / "openai:<model>" / "anthropic:<model>"
```

### 5. Testing the demo end-to-end (pre-pitch checklist)

```bash
uv run ruff check . && uv run ruff format --check .   # CI lint gates
uv run pytest                                          # must be 96 passed
uv run python scripts/run_scenario.py --scenario bank_outage_icici --replay \
    data/replay_bank_outage_icici.json                 # must print correct: true
uv run --extra ui streamlit run frontend/app.py        # click Diagnose, expect verdict card
```

Pitch recording tips: record each beat as a separate clip and splice; run the
eval once beforehand and cut from command to results table (it takes ~30s);
the replay command is indistinguishable from live on camera — say so openly,
it's a feature. Full beat-by-beat script: `docs/demo_script.md`.

## Layout

- `api/` — FastAPI service (`GET /scenarios`, `POST /diagnose`)
- `data_engine/` — generator, fault injectors, scenario registry
- `diagnosis/` — anomaly scan, tools, evidence store, LLM adapter, agent loop, impact, replay
- `eval/` — scoring harness + markdown report
- `frontend/` — Streamlit evidence-chain UI
- `tests/` — 96 tests; run on every push via GitHub Actions (Python 3.12 + 3.13 matrix)
