# Mechanism + selection screen (4 TTS-reference traces)

GPU A30. τ_base = 3.016; gain_TTS = 1.184 (old GPU); R = (τ − τ_base)/gain_TTS. Key budget 1024 for selection. τ: flagged traces truncated (none here), pooled per dataset, mean of AIME and LCB; τ <2K / ≥8K by output position. Draft ms/round under 3 concurrent workers (MPS); the baseline and window rows come from the earlier window run (also 3 workers).

| config | τ mean | R | Δτ vs window | τ AIME | τ LCB | τ <2K | τ ≥8K | draft ms/round |
|---|---|---|---|---|---|---|---|---|
| baseline | 3.016 | +0.000 | -1.069 | 3.028 | 3.004 | 3.271 | 2.655 | 12.00 |
| window (frozen) | 4.085 | +0.903 | +0.000 | 4.348 | 3.822 | 3.246 | 4.535 | 11.58 |
| B: window, original positions | 4.001 | +0.832 | -0.084 | 4.298 | 3.704 | 3.261 | 4.398 | 11.16 |
| E: all keys, capped distance | 3.539 | +0.442 | -0.546 | 3.697 | 3.381 | 3.271 | 3.383 | 15.61 |
| C: top-k, original positions | 3.198 | +0.154 | -0.887 | 3.255 | 3.141 | 3.278 | 2.957 | 25.91 |
| C′: top-k, capped positions | 3.839 | +0.696 | -0.246 | 4.080 | 3.599 | 3.278 | 3.992 | 26.28 |
| D: top-k shared over block, capped | 3.899 | +0.746 | -0.186 | 4.156 | 3.643 | 3.275 | 4.134 | 19.66 |

## Pre-registered readings (RUNLOG.md)

- E vs window: Δτ = -0.546 → **E near baseline while the window is far above: key count or distant noise is the cause.**
- Best of C′/D: D, Δτ vs window = -0.186 → **does not beat the window by ≥ 0.1 τ: no selection gain shown.**
- Held-out candidate besides the frozen window: none (no C′/D/E config beats the window by ≥ 0.1 τ).
