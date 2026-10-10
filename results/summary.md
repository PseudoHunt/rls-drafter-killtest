# Kill test summary: closed-form (RLS) test-time drafter adaptation vs TTS

## Decision

**RLS = STOP (futility rule §12.6).** On the dev set (flagged traces truncated, mean of AIME and LCB) every RLS and hidden-residual bias config is below the baseline: baseline τ = 3.028, best RLS (n0 = 128, λ = 0.9999) τ = 2.739, best hidden-residual bias (ρ = 0.999) τ = 2.304. Dev R of the best RLS config is negative, so Stage 4 (held-out test) was not run.

The "bias" arm below is the **hidden-residual bias** (exponentially weighted mean of h* − z, brief §5.2); it is not the KL/Reflex-form bias screened afterwards (see `results/screen.md`).

## Dev table

Decision variant: flagged traces truncated at loop onset (§11 A2, §12.4); τ pooled over rounds per dataset; mean = average of AIME and LCB. Timings were measured under MPS concurrency (not isolated).

### All 12 dev traces (11 after the drop)

TTS rows cover only the dev traces that finished before the grid was cancelled (n in brackets).

| method | config | τ AIME (n) | τ LCB (n) | τ mean | τ <2K | τ ≥8K | draft ms | update ms | solve ms | state MB |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | default | 3.259 (6) | 2.797 (5) | 3.028 | 3.316 | 2.667 | 10.26 | 0.02 | 0.00 | 0 |
| hidden-residual bias | rho=0.9 | 2.201 (6) | 1.954 (5) | 2.077 | 2.343 | 1.860 | 12.26 | 0.34 | 0.00 | 0 |
| hidden-residual bias | rho=0.99 | 2.438 (6) | 2.134 (5) | 2.286 | 2.535 | 2.058 | 12.70 | 0.35 | 0.00 | 0 |
| hidden-residual bias | rho=0.999 | 2.468 (6) | 2.140 (5) | 2.304 | 2.549 | 2.079 | 12.97 | 0.34 | 0.00 | 0 |
| rls | dtype=fp64_lam=0.995_n0=1024 | 2.724 (6) | 2.286 (5) | 2.505 | 2.654 | 2.270 | 22.87 | 5.84 | 13.93 | 150 |
| rls | dtype=fp64_lam=0.995_n0=128 | 2.769 (6) | 2.319 (5) | 2.544 | 2.665 | 2.319 | 23.25 | 5.48 | 14.06 | 150 |
| rls | dtype=fp64_lam=0.995_n0=16 | 2.768 (6) | 2.341 (5) | 2.555 | 2.641 | 2.335 | 22.56 | 5.88 | 14.88 | 150 |
| rls | dtype=fp64_lam=0.9999_n0=1024 | 2.935 (6) | 2.477 (5) | 2.706 | 2.721 | 2.512 | 22.96 | 5.87 | 14.22 | 150 |
| rls | dtype=fp64_lam=0.9999_n0=128 | 2.963 (6) | 2.515 (5) | 2.739 | 2.724 | 2.565 | 23.12 | 5.69 | 14.57 | 150 |
| rls | dtype=fp64_lam=0.9999_n0=16 | 2.859 (6) | 2.437 (5) | 2.648 | 2.589 | 2.531 | 23.30 | 5.24 | 13.99 | 150 |
| rls | dtype=fp64_lam=0.999_n0=1024 | 2.895 (6) | 2.422 (5) | 2.658 | 2.720 | 2.446 | 22.92 | 5.68 | 14.36 | 150 |
| rls | dtype=fp64_lam=0.999_n0=128 | 2.963 (6) | 2.510 (5) | 2.736 | 2.723 | 2.550 | 22.92 | 5.81 | 14.62 | 150 |
| rls | dtype=fp64_lam=0.999_n0=16 | 2.903 (6) | 2.478 (5) | 2.690 | 2.634 | 2.546 | 22.83 | 5.79 | 14.39 | 150 |
| rls | dtype=fp64_lam=1.0_n0=1024 | 2.923 (6) | 2.469 (5) | 2.696 | 2.724 | 2.494 | 13.88 | 2.06 | 1.58 | 100 |
| rls | dtype=fp64_lam=1.0_n0=128 | 2.920 (6) | 2.485 (5) | 2.703 | 2.714 | 2.503 | 14.06 | 2.07 | 1.58 | 100 |
| rls | dtype=fp64_lam=1.0_n0=16 | 2.822 (6) | 2.390 (5) | 2.606 | 2.581 | 2.474 | 13.68 | 2.04 | 1.56 | 100 |
| tts | lr=0.0001 | 3.596 (5) | – | – | – | – | 11.75 | 61.21 | 0.00 | 6150 |
| tts | lr=1e-05 | 4.487 (5) | – | – | – | – | 11.68 | 60.63 | 0.00 | 6150 |
| tts | lr=3e-05 | 4.425 (5) | 3.907 (2) | 4.166 | 3.436 | 4.653 | 12.15 | 62.24 | 0.00 | 6150 |

### TTS reference set: 4 dev traces (aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d)

All configs on exactly these 4 traces.

| method | config | τ AIME (n) | τ LCB (n) | τ mean | τ <2K | τ ≥8K | draft ms | update ms | solve ms | state MB |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | default | 3.027 (2) | 3.007 (2) | 3.017 | 3.275 | 2.654 | 11.28 | 0.02 | 0.00 | 0 |
| hidden-residual bias | rho=0.9 | 2.095 (2) | 2.036 (2) | 2.066 | 2.276 | 1.888 | 12.36 | 0.34 | 0.00 | 0 |
| hidden-residual bias | rho=0.99 | 2.328 (2) | 2.234 (2) | 2.281 | 2.484 | 2.094 | 12.94 | 0.35 | 0.00 | 0 |
| hidden-residual bias | rho=0.999 | 2.361 (2) | 2.251 (2) | 2.306 | 2.496 | 2.117 | 13.02 | 0.33 | 0.00 | 0 |
| rls | dtype=fp64_lam=0.995_n0=1024 | 2.596 (2) | 2.393 (2) | 2.495 | 2.606 | 2.266 | 22.56 | 6.16 | 14.82 | 150 |
| rls | dtype=fp64_lam=0.995_n0=128 | 2.646 (2) | 2.409 (2) | 2.527 | 2.611 | 2.283 | 23.04 | 5.28 | 14.01 | 150 |
| rls | dtype=fp64_lam=0.995_n0=16 | 2.633 (2) | 2.431 (2) | 2.532 | 2.553 | 2.296 | 22.34 | 5.80 | 14.91 | 150 |
| rls | dtype=fp64_lam=0.9999_n0=1024 | 2.825 (2) | 2.572 (2) | 2.699 | 2.661 | 2.484 | 22.97 | 5.81 | 14.42 | 150 |
| rls | dtype=fp64_lam=0.9999_n0=128 | 2.859 (2) | 2.588 (2) | 2.724 | 2.668 | 2.525 | 23.04 | 5.53 | 14.24 | 150 |
| rls | dtype=fp64_lam=0.9999_n0=16 | 2.773 (2) | 2.481 (2) | 2.627 | 2.536 | 2.481 | 22.69 | 5.48 | 14.37 | 150 |
| rls | dtype=fp64_lam=0.999_n0=1024 | 2.759 (2) | 2.533 (2) | 2.646 | 2.660 | 2.417 | 23.15 | 5.55 | 14.17 | 150 |
| rls | dtype=fp64_lam=0.999_n0=128 | 2.841 (2) | 2.602 (2) | 2.721 | 2.652 | 2.497 | 22.64 | 5.69 | 15.09 | 150 |
| rls | dtype=fp64_lam=0.999_n0=16 | 2.808 (2) | 2.560 (2) | 2.684 | 2.597 | 2.506 | 22.41 | 5.87 | 14.48 | 150 |
| rls | dtype=fp64_lam=1.0_n0=1024 | 2.810 (2) | 2.561 (2) | 2.686 | 2.666 | 2.464 | 13.94 | 2.10 | 1.61 | 100 |
| rls | dtype=fp64_lam=1.0_n0=128 | 2.796 (2) | 2.552 (2) | 2.674 | 2.663 | 2.456 | 14.15 | 2.09 | 1.59 | 100 |
| rls | dtype=fp64_lam=1.0_n0=16 | 2.726 (2) | 2.425 (2) | 2.576 | 2.535 | 2.407 | 13.54 | 2.03 | 1.55 | 100 |
| tts | lr=0.0001 | 3.637 (2) | – | – | – | – | 10.53 | 58.41 | 0.00 | 6150 |
| tts | lr=1e-05 | 4.557 (2) | – | – | – | – | 11.01 | 61.07 | 0.00 | 6150 |
| tts | lr=3e-05 | 4.495 (2) | 3.907 (2) | 4.201 | 3.336 | 4.626 | 11.55 | 60.89 | 0.00 | 6150 |

### R on the reference set

τ_base = 3.017, τ_TTS(lr=3e-5) = 4.201.

| arm | config | τ | R |
|---|---|---|---|
| hidden-residual bias (best on these 4) | rho=0.999 | 2.306 | -0.600 |
| RLS (best on these 4) | dtype=fp64_lam=0.9999_n0=128 | 2.724 | -0.248 |

Futility (§12.6): best RLS on all dev traces = dtype=fp64_lam=0.9999_n0=128 with τ = 2.739 < τ_base = 3.028 → dev R < 0 for any TTS that beats the baseline → **STOP**.

## Per-slot conditional acceptance P(a ≥ i | a ≥ i−1) — dev, flagged traces truncated

Counts pooled over rounds within a dataset; mean = average of AIME and LCB. A round counts toward slot i only if the (truncated) trace had a token at slot i.

| arm | slot 1 AIME | slot 1 LCB | slot 1 mean | slots 2–4 AIME | slots 2–4 LCB | slots 2–4 mean | slots 5–15 AIME | slots 5–15 LCB | slots 5–15 mean |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.723 | 0.701 | 0.712 | 0.677 | 0.612 | 0.645 | 0.690 | 0.604 | 0.647 |
| best RLS (n0=128, λ=0.9999) | 0.678 | 0.663 | 0.670 | 0.651 | 0.559 | 0.605 | 0.666 | 0.578 | 0.622 |
| best hidden-residual bias (ρ=0.999) | 0.623 | 0.584 | 0.604 | 0.567 | 0.484 | 0.525 | 0.612 | 0.519 | 0.566 |

Adaptation lowers acceptance at every slot group, most at slot 1 and slots 2–4; the hidden-residual bias costs the most. Together with the CPU diagnostic (RLS cuts out-of-sample hidden-state L2 error by ≈ 49% while lowering draft accuracy; HANDOFF.md), least squares on hidden states is not aligned with the drafter's argmax decisions.

## KL screen (A30, 4 TTS-reference traces) = STOP

Pre-registered rule (RUNLOG.md, A30 session): best arm R ≥ 0.5 promising, R < 0.25 stop. KL-bias (ρ = 0.9,
α ∈ {0.005, 0.01, 0.025}) and KL-linear (Adam lr ∈ {1e-7, 1e-6, 1e-5}), fp32. A30 baseline τ = 3.016; gain_TTS = 1.184
(old GPU).

**Best arm: KL-linear lr = 1e-6, τ = 3.162, R = +0.12 → STOP** (optimistic: tuned on the same 4 traces). Best
KL-bias: α = 0.01, R = +0.02; α = 0.025 is below the baseline (R = −0.12). Measured cost: 7–9 ms per round of
update (vs ≈ 11 ms drafting), plus a 1.45 GiB fp32 copy of the lm_head weight; KL-linear state 75 MB. Full table:
`results/screen.md`.
