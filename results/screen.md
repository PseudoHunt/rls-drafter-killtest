# Screen: KL-bias and KL-linear vs baseline (4 TTS-reference traces)

Traces: aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d. GPU: A30. τ = committed tokens/round, flagged traces truncated, pooled per dataset, mean of AIME and LCB.

- τ_base (A30) = 3.016. A30 baseline τ = 3.016 vs old-GPU 3.017 (-0.04%) — within 1%.
- gain_TTS = 1.184 (old GPU: τ_TTS(lr=3e-5) − τ_base = 4.201 − 3.017 = 1.184); R = (τ − τ_base,A30) / gain_TTS.
- Thresholds: R = 0.5 ↔ τ = 3.608; R = 0.25 ↔ τ = 3.312.
- Tuned on the same 4 traces, so the best arm's R is optimistic.

| method | config | τ AIME | τ LCB | τ mean | R | update ms/round | draft ms/round | state MB | KL-linear rounds with ‖Az̃‖/‖z‖ > 0.2 (max) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | default | 3.028 | 3.004 | 3.016 | +0.000 | 0.02 | 10.99 | 0.00 | – |
| klbias | alpha=0.005_rho=0.9 | 3.041 | 3.041 | 3.041 | +0.022 | 7.51 | 13.84 | 0.01 | – |
| klbias | alpha=0.01_rho=0.9 | 3.045 | 3.043 | 3.044 | +0.024 | 7.50 | 14.84 | 0.01 | – |
| klbias | alpha=0.025_rho=0.9 | 2.850 | 2.902 | 2.876 | -0.118 | 7.27 | 14.60 | 0.01 | – |
| kllinear | lr=1e-05 | 3.137 | 3.042 | 3.090 | +0.062 | 8.36 | 13.36 | 75.03 | 34 (0.209) **FLAG** |
| kllinear | lr=1e-06 | 3.185 | 3.138 | 3.162 | +0.123 | 8.82 | 14.22 | 75.03 | 0 (0.076) |
| kllinear | lr=1e-07 | 3.110 | 3.080 | 3.095 | +0.067 | 8.14 | 15.22 | 75.03 | 0 (0.026) |

Timings were measured under concurrent workers (MPS), not isolated. State MB excludes the shared fp32 copy of the lm_head weight (1.45 GiB on Qwen3-4B) that both KL arms use for W.

## Decision (pre-registered, RUNLOG.md)

Best arm: kllinear lr=1e-06, τ = 3.162, R = +0.123 → **stop this line**.
