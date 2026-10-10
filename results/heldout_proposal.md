# Held-out test proposal (GATE — not launched; needs approval)

Frozen now (2026-10-10, after the mechanism screen): **window W = 1024, keep prompt** (NOTES.md §9; contiguous
positions). No C′/D/E config beat it by ≥ 0.1 τ on the 4 reference traces, so no extra arm.

## Configs
1. baseline — all 50 test traces (24 AIME, 26 LCB; 1 dropped by the loop rule → 49 in the metric).
2. window W = 1024, keep prompt — all test traces.
3. TTS lr = 3e-5 — all test traces (re-measured on the A30, so R uses an A30 TTS gain, not the old-GPU one).
4. window + TTS lr = 3e-5 — 16 test traces (8 AIME + 8 LCB, drawn with a fixed seed and listed in RUNLOG before
   the run). **Needs code first**: TTS's gradient forward reads the stock cache (`PrefixCache`); it needs a read-only
   WindowCache path + CPU tests (≈ 30–45 min of coding; can be done on a CPU instance to save A30 time).

Metric: decision variant (flagged traces truncated, mean of AIME and LCB), also exclusion and loops-included as
robustness; τ <2K / ≥8K; drafting and update ms/round.

**Decision rule:** window R = (τ_window − τ_base)/(τ_TTS − τ_base) on test ≥ 0.5 → confirms the headline. (Also
report window + TTS vs TTS on the 16 traces, descriptively.)

## A30-hour projection (24 GB: 1 TTS worker + 1 light worker; TTS jobs serialised)

Test outputs: 660,974 tokens (replays run on full traces; truncation is applied afterwards).

| job | rounds (≈ tokens/τ) | ms/round | worker-h |
|---|---|---|---|
| baseline | 661K / 3.0 ≈ 220K | ≈ 12 | 0.7 |
| window | 661K / 4.0 ≈ 165K | ≈ 12 | 0.55 |
| TTS lr = 3e-5 (50 traces) | 661K / 4.2 ≈ 157K | 85–100 (A30 probe: 62 early in a trace; old GPU 85 over a full trace) | 3.7–4.4 |
| window + TTS (16 traces) | ≈ 212K / 5 ≈ 42K | ≈ 75 | ≈ 0.9 |
| teacher-forced passes, model loads, report | | | ≈ 0.3 |

TTS jobs (≈ 4.6–5.3 h) are the critical path; baseline and window run alongside on the light worker.
**Projection: ≈ 5–6 A30-h wall for the run, + ≈ 0.5 A30-h if the window + TTS code is written on the instance;
range 4.5–7 A30-h** (main uncertainty: A30 TTS cost at long context).
