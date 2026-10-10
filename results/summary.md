# Kill test summary: closed-form (RLS) test-time drafter adaptation vs TTS

<!-- HELDOUT BEGIN -->

## Held-out test (A30, 49 test traces) — leads with S2 and the window

- **Window (W 1024 + prompt): R = 0.718** [0.688, 0.743]; window − base +0.795 [+0.695, +0.870].
- S2: R = 0.874 [0.845, 0.900].
- Headline: window R ≥ 0.5: R = 0.718 → **YES**
- S2 adds value: S2 − window ≥ 0.05 with CI > 0, and CIs of S2 − C1, S2 − C2 exclude 0 (above): +0.173 [+0.150, +0.195]; C1 +0.106 [+0.089, +0.123]; C2 +0.110 [+0.092, +0.127] → **YES**
- S2 dominates TTS: lower CI bound of S2 − TTS(stride 1) ≥ 0: -0.139 [-0.173, -0.107] → **NO**
- Hybrid: hybrid − TTS(stride 1) CI above 0, at ≤ 10% of TTS(stride 1)'s update ms/round (16 traces): +0.130 [+0.063, +0.176]; update ms ratio = 43.0/69.6 = 61.8% → **NO**
- Residual after S2 (report only): +0.491 [+0.421, +0.558]

Full report: `results/heldout.md` (stage: done).

<!-- HELDOUT END -->

<!-- VPA BEGIN -->

## Verifier-Prior Attention (VPA; dev-tuned, stage: not promoted)

- Screen (4 reference traces): best = M=64_beta=0.0_lam=0.0, τ = 4.172 (S2 4.231, window 4.085).
- Confirmation (7 dev traces): VPA − S2 = -0.049 (need ≥ +0.05), VPA ≥ S2 on 1/7 (need ≥ 75%) → **not promoted**.

Full report: `results/vpa.md`.

<!-- VPA END -->
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

## Windowed drafter context (A30, 4 TTS-reference traces) = promising

Training-free: the drafter attends to a kept set + the last W committed context positions, contiguous position ids
inside the kept cache (StreamingLLM-style; NOTES.md §9). Target and verification unchanged. Pre-registered in RUNLOG.md.

| config | τ mean | R | τ <2K | τ ≥8K | draft ms/round |
|---|---|---|---|---|---|
| baseline | 3.016 | 0 | 3.271 | 2.655 | 12.00 |
| W = 1024, keep 64 | 4.037 | +0.862 | 3.230 | 4.483 | 10.73 |
| **W = 1024, keep prompt** | **4.085** | **+0.903** | 3.246 | 4.535 | 11.58 |
| W = 4096, keep 64 | 4.015 | +0.844 | 3.271 | 4.436 | 12.44 |
| W = 4096, keep prompt | 3.996 | +0.828 | 3.271 | 4.383 | 12.55 |

**Best R = +0.90 → promising** (pre-registered: plan a full held-out test). τ for output positions < 2K is unchanged
(the window is barely active there), while τ ≥ 8K rises from 2.655 to 4.4–4.5: the long-output decay on these traces
is mostly a long-context effect inside the drafter. All 4 configs land at R = 0.83–0.90, so the selection optimism
from tuning on the same 4 traces is small; the evidence is still only 4 traces (2 AIME, 2 LCB). gain_TTS is the
old-GPU TTS gain, used as a yardstick. Full table: `results/window.md`.

## Mechanism + selection screen (A30, 4 TTS-reference traces)

Training-free context policies (NOTES.md §10), key budget 1024 where keys are selected. Pre-registered in RUNLOG.md.

| config | τ | R | Δτ vs window | τ AIME | τ LCB | τ <2K | τ ≥8K | draft ms/round* |
|---|---|---|---|---|---|---|---|---|
| baseline | 3.016 | 0 | −1.069 | 3.028 | 3.004 | 3.271 | 2.655 | 12.0 |
| **window W = 1024, keep prompt (frozen)** | **4.085** | **+0.90** | 0 | 4.348 | 3.822 | 3.246 | 4.535 | 11.6 |
| B: window, original position ids | 4.001 | +0.83 | −0.084 | 4.298 | 3.704 | 3.261 | 4.398 | 11.2 |
| E: all keys, distance capped at 4096 | 3.539 | +0.44 | −0.546 | 3.697 | 3.381 | 3.271 | 3.383 | 15.6 |
| C: top-1024, original positions | 3.198 | +0.15 | −0.887 | 3.255 | 3.141 | 3.278 | 2.957 | 25.9 |
| C′: top-1024, capped positions | 3.839 | +0.70 | −0.246 | 4.080 | 3.599 | 3.278 | 3.992 | 26.3 |
| D: top-1024 shared over block, capped | 3.899 | +0.75 | −0.186 | 4.156 | 3.643 | 3.275 | 4.134 | 19.7 |

\* under 2–3 concurrent workers; indicative only.

Readings (pre-registered): E recovers 49% of the window's gain — **neither** "E ≈ window" (distance is the main cause)
nor "E ≈ baseline" applies cleanly. Together with B ≈ window (−0.08; original positions, i.e. the full query↔prompt
distance, cost little once the middle is dropped), the decay is driven mostly by **attending to many distant middle
keys**, with the out-of-range distance a secondary part (capping it alone recovers half). Selection does **not**
beat the window: best D −0.19, C′ −0.25 (C, which selects under the original rotation, barely helps) → no selection
gain shown; CrossCov-U is not indicated. Held-out candidate besides the frozen window: none.

## T0b: target-guided far-context selection (A30, 4 TTS-reference traces) = adds value → S2 frozen

Visible context = prompt + window + 16 far 32-token chunks chosen every round by the target's head-summed attention
row at the last committed position t'−1 (guarded), renumbered contiguously (NOTES.md §11). Pre-registered in RUNLOG.md.

| config | τ | R | Δτ vs window | τ AIME | τ LCB | τ <2K | τ ≥8K | draft ms/round* | visible keys |
|---|---|---|---|---|---|---|---|---|---|
| window W 1024 + prompt | 4.085 | +0.90 | 0 | 4.348 | 3.822 | 3.246 | 4.535 | 11.6 | 1310 |
| S1: + 16 far chunks, target layer 35 | 4.192 | +0.99 | +0.107 | 4.476 | 3.908 | 3.255 | 4.696 | 9.2 | 1754 |
| **S2: + 16 far chunks, target layers 17/25/33** | **4.231** | **+1.03** | **+0.146** | 4.533 | 3.929 | 3.256 | 4.779 | 8.4 | 1754 |
| S3: W 512 + 16 far chunks (window budget) | 4.157 | +0.96 | +0.072 | 4.438 | 3.875 | 3.249 | 4.618 | 8.3 | 1308 |
| C1: window W 1536 + prompt (S1 budget) | 4.124 | +0.94 | +0.039 | 4.385 | 3.863 | 3.264 | 4.587 | 8.3 | 1757 |
| C2: W 1024 + 16 random far chunks | 4.134 | +0.94 | +0.049 | 4.407 | 3.860 | 3.264 | 4.594 | 8.8 | 1756 |

\* window row from a 3-worker run, others 2 workers; indicative only.

**Rule met:** best S = S2 beats the window by +0.146 (≥ 0.1) and C1 (+0.107) and C2 (+0.097). Per trace, S2 beats
the window, C1 and C2 on all 4 traces (+0.05 to +0.21). Caveats: margins over the controls are ≈ 0.1 τ (2–3%); S2 is
the best of 3 scored variants on the same 4 traces; C2 is a single random seed. At equal budget (S3 vs window) the
gain is +0.07. Extra target-side cost per round (A30, unfused upper bound): 0.6–1.1 ms for S2's 3 layers at 2K–16K
context, 0.2–0.4 ms for one layer (results/attn_row_cost.md). **S2 frozen** as an arm in T1 and T2.
