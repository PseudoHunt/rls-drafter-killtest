# Windowed drafter context (StreamingLLM-style) vs baseline (4 TTS-reference traces)

Traces: aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d. GPU: A30. τ = committed tokens/round, flagged traces truncated, pooled per dataset, mean of AIME and LCB.

- τ_base (A30) = 3.016. A30 baseline τ = 3.016 vs old-GPU 3.017 (-0.04%) — within 1%.
- gain_TTS = 1.184 (old GPU: τ_TTS(lr=3e-5) − τ_base = 4.201 − 3.017 = 1.184); R = (τ − τ_base,A30) / gain_TTS.
- Thresholds: R = 0.5 ↔ τ = 3.608; R = 0.25 ↔ τ = 3.312.
- Tuned on the same 4 traces, so the best arm's R is optimistic.

| method | config | τ AIME | τ LCB | τ mean | R | τ <2K | τ ≥8K | update ms/round | draft ms/round | state MB |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | default | 3.028 | 3.004 | 3.016 | +0.000 | 3.271 | 2.655 | 0.02 | 12.00 | 0.00 |
| window | W=1024_keep=64 | 4.319 | 3.755 | 4.037 | +0.862 | 3.230 | 4.483 | 0.02 | 10.73 | 0.00 |
| window | W=1024_keep=prompt | 4.348 | 3.822 | 4.085 | +0.903 | 3.246 | 4.535 | 0.02 | 11.58 | 0.00 |
| window | W=4096_keep=64 | 4.316 | 3.713 | 4.015 | +0.844 | 3.271 | 4.436 | 0.02 | 12.44 | 0.00 |
| window | W=4096_keep=prompt | 4.303 | 3.689 | 3.996 | +0.828 | 3.271 | 4.383 | 0.02 | 12.55 | 0.00 |

Timings were measured under concurrent workers (MPS), not isolated; τ <2K / ≥8K = output positions.

## Decision (pre-registered, RUNLOG.md)

Best arm: window W=1024_keep=prompt, τ = 4.085, R = +0.903 → **promising**.
