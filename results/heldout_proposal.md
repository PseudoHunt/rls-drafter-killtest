# Held-out test proposal (GATE — not launched; needs approval)

Frozen now (2026-10-10, after the mechanism screen): **window W = 1024, keep prompt** (NOTES.md §9; contiguous
positions). No C′/D/E config beat it by ≥ 0.1 τ on the 4 reference traces, so no extra arm from T0.

**Update after T0b (2026-10-10 10:37 UTC):** target-guided selection met its pre-registered rule → **S2 frozen**
(W 1024 + prompt + 16 far 32-token chunks, scored by the mean of target layers 17/25/33 at t'−1; NOTES.md §11) and
added as an arm in T1 and T2 alongside the window.

## Configs
T1 (all 50 test traces: 24 AIME, 26 LCB; 1 dropped by the loop rule → 49 in the metric):
1. baseline.
2. window W = 1024, keep prompt.
3. **S2** (target-guided far chunks; the teacher-forced pass also computes the target attention summaries).
4. TTS lr = 3e-5 (re-measured on the A30, so R uses an A30 TTS gain, not the old-GPU one).
T2 (16 test traces: 8 AIME + 8 LCB, drawn with a fixed seed and listed in RUNLOG before the run):
5. window + TTS lr = 3e-5.
6. **S2 + TTS lr = 3e-5.**
**Needs code first** for 5–6: TTS's gradient forward reads the stock cache (`PrefixCache`); it needs a read-only
WindowCache path (window and far chunks) + CPU tests (≈ 45–60 min of coding; can be done on a CPU instance to save
A30 time).

Metric: decision variant (flagged traces truncated, mean of AIME and LCB), also exclusion and loops-included as
robustness; τ <2K / ≥8K; drafting and update ms/round.

**Decision rule:** window R = (τ_window − τ_base)/(τ_TTS − τ_base) on test ≥ 0.5 → confirms the headline. (Also
report window + TTS vs TTS on the 16 traces, descriptively.) **S2 needs its own rule before T1 — proposal:** S2
counts as better than the window on test if τ_S2 − τ_window ≥ 0.1 (mean of AIME and LCB) and it is positive on both
datasets; otherwise report descriptively.

## A30-hour projection (24 GB: 1 TTS worker + 1 light worker; TTS jobs serialised)

Test outputs: 660,974 tokens (replays run on full traces; truncation is applied afterwards).

| job | rounds (≈ tokens/τ) | ms/round | worker-h |
|---|---|---|---|
| baseline | 661K / 3.0 ≈ 220K | ≈ 12 | 0.7 |
| window | 661K / 4.0 ≈ 165K | ≈ 12 | 0.55 |
| TTS lr = 3e-5 (50 traces) | 661K / 4.2 ≈ 157K | 85–100 (A30 probe: 62 early in a trace; old GPU 85 over a full trace) | 3.7–4.4 |
| S2 | 661K / 4.2 ≈ 157K | ≈ 12 | 0.55 |
| window + TTS (16 traces) | ≈ 212K / 5 ≈ 42K | ≈ 75 | ≈ 0.9 |
| S2 + TTS (16 traces) | ≈ 212K / 5 ≈ 42K | ≈ 75 | ≈ 0.9 |
| teacher-forced passes (+ attention summaries), model loads, report | | | ≈ 0.4 |

TTS jobs (≈ 5.5–6.2 h) are the critical path; baseline, window and S2 run alongside on the light worker.
**Projection (updated after T0b): ≈ 6–7 A30-h wall for the run, + ≈ 0.75 A30-h if the window/S2 + TTS code is
written on the instance; range 5.5–8 A30-h** (main uncertainty: A30 TTS cost at long context).
