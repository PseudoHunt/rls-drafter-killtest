# Held-out test (A30) — stage: done

Pre-registered in RUNLOG.md (2026-10-10). Decision variant: flagged traces truncated at loop onset, dropped traces excluded; τ = mean over AIME and LCB of the pooled τ. CIs: paired bootstrap over traces (stratified by dataset), B = 10000, seed 0, 95% percentile. TTS setting (T0): lr = 1e-5, stale cache (best stale lr on the AIME dev grid: 1e-5 4.487, 3e-5 4.425, 1e-4 3.596 over 5 traces; refresh check not run).

## τ per arm

| arm | traces | τ | τ AIME | τ LCB | τ <2K | τ ≥8K | τ excl. flagged | τ loops incl. |
|---|---|---|---|---|---|---|---|---|
| baseline | 49 | 3.100 | 3.314 | 2.887 | 3.516 | 2.588 | 3.046 | 3.443 |
| window | 49 | 3.896 | 4.109 | 3.682 | 3.507 | 4.100 | 3.904 | 4.507 |
| S2 | 49 | 4.069 | 4.276 | 3.862 | 3.516 | 4.416 | 4.088 | 4.704 |
| C1 | 49 | 3.963 | 4.165 | 3.761 | 3.516 | 4.202 | 3.976 | 4.585 |
| C2 | 49 | 3.959 | 4.168 | 3.749 | 3.512 | 4.191 | 3.971 | 4.578 |
| TTS | 49 | 4.208 | 4.353 | 4.063 | 3.693 | 4.483 | 4.209 | 4.874 |
| TTS stride 10 | 16 | 3.670 | 3.881 | 3.459 | 3.478 | 3.656 | 3.674 | 4.138 |
| hybrid | 16 | 4.429 | 4.609 | 4.250 | 3.513 | 5.191 | 4.513 | 5.046 |
| S2+TTS | 16 | 4.610 | 4.787 | 4.434 | 3.641 | 5.317 | 4.688 | 5.248 |

## Contrasts (paired bootstrap 95% CI)

| contrast | traces | Δτ [95% CI] |
|---|---|---|
| window − base | 49 | +0.795 [+0.695, +0.870] |
| S2 − base | 49 | +0.968 [+0.855, +1.058] |
| TTS − base | 49 | +1.108 [+0.987, +1.208] |
| S2 − window | 49 | +0.173 [+0.150, +0.195] |
| S2 − C1 | 49 | +0.106 [+0.089, +0.123] |
| S2 − C2 | 49 | +0.110 [+0.092, +0.127] |
| S2 − TTS(stride 1) | 49 | -0.139 [-0.173, -0.107] |
| hybrid − TTS(stride 1) | 16 | +0.130 [+0.063, +0.176] |
| (S2 + TTS stride 1) − S2 | 16 | +0.491 [+0.421, +0.558] |
| TTS stride 10 − TTS stride 1 | 16 | -0.629 [-0.717, -0.495] |
| R_window = gain_window / gain_TTS | 49 | +0.718 [+0.688, +0.743] |
| R_S2 = gain_S2 / gain_TTS | 49 | +0.874 [+0.845, +0.900] |

## Pre-registered claims

- **Headline: window R ≥ 0.5**: R = 0.718 → **YES**
- **S2 adds value: S2 − window ≥ 0.05 with CI > 0, and CIs of S2 − C1, S2 − C2 exclude 0 (above)**: +0.173 [+0.150, +0.195]; C1 +0.106 [+0.089, +0.123]; C2 +0.110 [+0.092, +0.127] → **YES**
- **S2 dominates TTS: lower CI bound of S2 − TTS(stride 1) ≥ 0**: -0.139 [-0.173, -0.107] → **NO**
- **Hybrid: hybrid − TTS(stride 1) CI above 0, at ≤ 10% of TTS(stride 1)'s update ms/round (16 traces)**: +0.130 [+0.063, +0.176]; update ms ratio = 43.0/69.6 = 61.8% → **NO**
- **Residual after S2 (report only)**: +0.491 [+0.421, +0.558]

## Costs

TTS variants on the 16 T2 traces (one TTS process alone on the GPU):

| variant | context | update ms/round | update ms per gradient round | draft ms/round | rounds |
|---|---|---|---|---|---|
| TTS | full | 69.60 | 69.60 | 7.35 | 40747 |
| TTS stride 10 | full | 7.99 | 71.12 | 7.25 | 48357 |
| hybrid | S2 | 43.01 | 43.01 | 8.18 | 39512 |
| S2+TTS | S2 | 56.24 | 56.24 | 8.12 | 37988 |

S2 extra target-side cost (one query row against the K cache of target layers 17/25/33, bytes read): 3 layers × 8 KV heads × 128 × 2 B × t'−1 keys = 41.2 MB per round on average over test rounds (mean t'−1 = 6709) → ≈ 0.044 ms per round at the A30's 933 GB/s; at 16K context 100.7 MB ≈ 0.108 ms (microbenchmark, unfused: results/attn_row_cost.md).

Drafting ms/round, isolated (one process alone on the GPU, 2 dev traces):

| arm | draft ms/round | rounds |
|---|---|---|
| baseline | 7.30 | 7711 |
| window | 7.82 | 5680 |
| S2 | 8.04 | 5521 |
