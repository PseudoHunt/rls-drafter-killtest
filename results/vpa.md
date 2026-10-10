# Verifier-Prior Attention (VPA) — stage: not promoted

Pre-registered: RUNLOG.md addendum B. Kept context = prompt + last W0 = 1024 + top M far 32-token chunks (S2's signal); bias on kept context keys b_j = β·log(m_j + 1/N_ctx) − λ·log(1 + max(0, d_j − W0)/W0) (distance term on far keys only), centred; block keys unbiased. τ: decision variant, mean of AIME and LCB pooled τ. Draft ms under 2 concurrent workers (indicative).

## Step 1 — screen (4 reference traces), M = 64

| config | τ | τ AIME | τ LCB | τ ≥8K | draft ms/round |
|---|---|---|---|---|---|
| baseline | 3.016 | 3.028 | 3.004 | 2.655 | 12.00 |
| window (W 1024 + prompt) | 4.085 | 4.348 | 3.822 | 4.535 | 11.58 |
| S2 (M = 16, no bias) | 4.231 | 4.533 | 3.929 | 4.779 | 8.38 |
| VPA M=64_beta=0.0_lam=0.0 | 4.172 | 4.476 | 3.869 | 4.664 | 10.23 |
| VPA M=64_beta=0.5_lam=0.0 | 3.528 | 3.668 | 3.388 | 3.723 | 12.97 |
| VPA M=64_beta=0.5_lam=0.5 | 3.414 | 3.517 | 3.311 | 3.565 | 13.03 |
| VPA M=64_beta=0.5_lam=1.0 | 3.245 | 3.305 | 3.184 | 3.325 | 12.88 |
| VPA M=64_beta=1.0_lam=0.0 | 1.865 | 1.838 | 1.892 | 1.836 | 12.94 |
| VPA M=64_beta=1.0_lam=0.5 | 1.823 | 1.796 | 1.849 | 1.800 | 13.11 |
| VPA M=64_beta=1.0_lam=1.0 | 1.766 | 1.741 | 1.791 | 1.743 | 12.57 |

Chosen (best mean τ): **M=64_beta=0.0_lam=0.0**.

## Step 2 — confirmation (7 non-dropped dev traces outside the screen set)

| config | τ | τ AIME | τ LCB | τ ≥8K | draft ms/round |
|---|---|---|---|---|---|
| window | 3.898 | 4.076 | 3.720 | 4.370 | 9.29 |
| S2 | 4.070 | 4.235 | 3.906 | 4.637 | 8.63 |
| VPA M=64_beta=0.0_lam=0.0 | 4.021 | 4.203 | 3.839 | 4.526 | 9.47 |

| trace | window τ | S2 τ | VPA τ | VPA ≥ S2 |
|---|---|---|---|---|
| aime24_II_02 | 4.047 | 4.209 | 4.192 | no |
| aime24_II_09 | 3.369 | 3.383 | 3.386 | yes |
| aime24_II_13 | 4.550 | 4.759 | 4.707 | no |
| aime24_I_13 | 4.064 | 4.260 | 4.214 | no |
| lcb_3701 | 3.328 | 3.513 | 3.475 | no |
| lcb_arc193_d | 3.776 | 3.961 | 3.887 | no |
| lcb_arc195_c | 4.149 | 4.330 | 4.232 | no |

Promotion rule: VPA − S2 = -0.049 (need ≥ +0.05), VPA ≥ S2 on 1/7 (need ≥ 75%) → **not promoted**

Not promoted: dev results reported as is. The window (M = 0, β = λ = 0) and S2 (M = 16, β = λ = 0) are limits of the VPA family; on dev the hard-selection limit performed best.
