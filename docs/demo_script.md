# RootLens — 5-minute pitch script

> Record at 1080p, terminal + Streamlit side by side. One live LLM run on camera;
> everything else replayed from cache so the demo cannot fail live.

## Beat 1 — The problem (0:00–0:45)

"At 2pm, a payment platform's success rate drops. An ops analyst spends 45 minutes
pulling dashboards and slicing dimensions to figure out why. Was it an issuer bank?
A card network? A risk rule that silently started declining high-ticket payments?
A retry storm? Today that diagnosis is manual, slow, and guesswork-prone."

**Show:** the anomaly scan output for `bank_outage_icici` — the deterministic,
no-LLM stage that finds `issuer_bank=ICICI: 97.5% -> 12.4% success` in seconds.

Key line: **"The LLM never invents numbers. The scan finds the signal; the LLM only
explains it."**

## Beat 2 — Why this isn't a ChatGPT wrapper (0:45–1:45)

1. **Deterministic first.** A segmented anomaly scan runs before the LLM sees
   anything — every dimension, ranked by severity, statistically gated.
2. **Falsification, not vibes.** The agent is *required* to state what would disprove
   each hypothesis and check it (show the issuer×network matrix tool call).
3. **Full audit trail.** Every claim cites a `call_id` — open the evidence chain in
   the UI and click through to the exact SQL and rows behind the verdict.
4. **Honest uncertainty.** If the hypothesis budget runs out, RootLens says
   "inconclusive — here's what data is missing" instead of guessing.

**Show:** Streamlit verdict card → expand `call_002` (compare_segments) → the verdict
text references exactly what that query returned.

## Beat 3 — Measured, not demoed (1:45–3:15)

"Anyone can demo three cherry-picked examples. We measure diagnosis accuracy against
injected faults with known ground truth, reported per difficulty tier — never blended."

**Show:** `python scripts/run_eval.py --agent rule` live — 14 scenarios, tier table.

Numbers to quote (rule baseline, current build):
- clean tier: 9/9 = 100% (including the healthy false-positive control and the
  benign-spike control — staying *silent* is scored as a pass)
- compound: 1/1, noisy: 2/2, red-herring: 2/2 (the agent blames the real outage,
  not the correlated marketing spike)
- false positives: 0 across all controls

Then: **"The leaderboard is model-agnostic. The rule baseline costs $0 and answers in
milliseconds — any LLM we put on the harness has to beat it on accuracy to justify its
cost and latency. Same scenarios, same scoring, per-tier, honest."

## Beat 4 — One live run (3:15–4:15)

Run one genuinely live diagnosis (`--provider anthropic --record`) to prove the harness
isn't canned, then note: "we keep exactly one live run per demo; everything else replays
from cached transcripts so a demo never depends on an API call succeeding."

If the live run is inconclusive or wrong: **show it anyway.** Honest failure reporting
is the product thesis. (Backup: the cached transcript replays the correct diagnosis.)

## Beat 5 — Impact + close (4:15–5:00)

- Business impact translation: time-to-diagnosis in seconds vs the documented 45-minute
  manual baseline; GMV-at-risk shown as an explicit upper bound, not a inflated claim.
- Why now / why us: payments observability where every number is traceable — the audit
  trail and eval harness are the parts a wrapper cannot replicate.
- Close: "RootLens: evidence before hypotheses, falsification before confirmation,
  measurement before claims."

## Demo-day checklist

- [ ] `uv sync && uv run pytest` green on camera (fast, ~5s)
- [ ] replay caches pre-generated for demo scenarios
- [ ] one fresh live run kept un-cached for Beat 4
- [ ] `eval/results/*_leaderboard.md` numbers current in this doc
- [ ] no API keys on screen; `.env` never printed
