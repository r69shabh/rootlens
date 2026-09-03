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

```bash
uv sync
uv run pytest                      # 45 tests
uv run python scripts/run_scenario.py --list                          # all 14 scenarios
uv run python scripts/run_scenario.py --scenario bank_outage_icici    # deterministic proof
uv run python scripts/run_scenario.py --scenario bank_outage_icici \
    --provider anthropic --record data/replay.json --dump d.json       # live run, cached
uv run python scripts/run_scenario.py --scenario bank_outage_icici \
    --replay data/replay.json                                          # offline demo
uv run --extra ui streamlit run frontend/app.py                        # evidence-chain UI
```

## Layout

- `data_engine/` — generator, fault injectors, scenario registry
- `diagnosis/` — anomaly scan, tools, evidence store, LLM adapter, agent loop, impact, replay
- `eval/` — scoring harness + markdown report
- `frontend/` — Streamlit evidence-chain UI
- `tests/` — 45 unit tests; run on every push via GitHub Actions
