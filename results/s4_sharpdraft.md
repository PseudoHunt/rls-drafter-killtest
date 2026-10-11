# Session 4, item 1 — SharpDraft (49 test traces, A30)

SharpDraft (arXiv 2610.05106 Eq. 5/18; k 32, s 5, cap e^0.5, N0 2048 = the paper's fixed Qwen value, not tuned). SD = released drafter, full context. SD+S2+cal = frozen S2+cal checkpoint in the S2 layout: **LCB rounds from the new run (26 traces); AIME rounds are the S2+cal replay, identical by construction** (β̂ = 1 on every AIME round: N ≤ 1,965 ≤ N0; tests/test_sharpdraft.py). Update ms/round = 0 for both (one scalar per round).

| arm | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | update ms/round | draft ms/round* | rounds |
|---|---|---|---|---|---|---|---|---|
| TTS stride 1 | 4.208 | 3.693 | 4.396 | 4.213 | 4.483 | 69.63 | 7.4 | 135679 |
| S2+cal | 4.408 | 3.779 | 4.370 | 4.578 | 4.756 | 0.00 | 7.9 | 130327 |
| SharpDraft | 3.644 | 3.517 | 3.980 | 3.630 | 3.557 | 0.02 | 9.7 | 161764 |
| SD+S2+cal | 4.410 | 3.781 | 4.369 | 4.579 | 4.761 | 0.02 | 8.9 | 130240 |

\* draft ms under 2 concurrent light workers for the new arms: indicative only.

## Contrasts (paired 95% CI)

| contrast | overall | 0–2K | 2–4K | 4–8K | ≥8K | AIME | LCB |
|---|---|---|---|---|---|---|---|
| SharpDraft − TTS stride 1 | -0.564 [-0.624, -0.502] | -0.176 [-0.198, -0.154] | -0.416 [-0.473, -0.366] | -0.583 [-0.655, -0.513] | -0.925 [-1.039, -0.823] | -0.464 [-0.537, -0.380] | -0.664 [-0.759, -0.575] |
| SharpDraft − S2+cal | -0.764 [-0.824, -0.702] | -0.262 [-0.282, -0.242] | -0.390 [-0.424, -0.357] | -0.947 [-1.024, -0.879] | -1.198 [-1.318, -1.096] | -0.718 [-0.790, -0.636] | -0.810 [-0.904, -0.720] |
| SD+S2+cal − TTS stride 1 | +0.202 [+0.181, +0.224] | +0.088 [+0.066, +0.110] | -0.027 [-0.068, +0.011] | +0.366 [+0.326, +0.415] | +0.279 [+0.238, +0.338] | +0.254 [+0.225, +0.288] | +0.151 [+0.120, +0.177] |
| SD+S2+cal − S2+cal | +0.002 [+0.001, +0.004] | +0.001 [-0.000, +0.003] | -0.001 [-0.006, +0.005] | +0.002 [-0.001, +0.005] | +0.006 [+0.002, +0.010] | +0.000 [+0.000, +0.000] | +0.005 [+0.001, +0.008] |

## Per dataset (95% CI, bootstrap within the dataset)

| arm | τ AIME | τ LCB |
|---|---|---|
| TTS stride 1 | 4.353 [4.151, 4.561] (n=23) | 4.063 [3.943, 4.188] (n=26) |
| S2+cal | 4.607 [4.384, 4.838] (n=23) | 4.209 [4.091, 4.330] (n=26) |
| SharpDraft | 3.889 [3.701, 4.102] (n=23) | 3.399 [3.323, 3.483] (n=26) |
| SD+S2+cal | 4.607 [4.384, 4.838] (n=23) | 4.214 [4.097, 4.333] (n=26) |

## Further contrasts (paired 95% CI)

| contrast | CI |
|---|---|
| SharpDraft − baseline | +0.544 [+0.478, +0.594] |
| SharpDraft − S2 | -0.425 [-0.467, -0.374] |
| SD+S2+cal − SharpDraft | +0.766 [+0.705, +0.826] |

## Readings (pre-registered)

- SD vs S2+cal: -0.764 [-0.824, -0.702] → SD **below** S2+cal.
- SD+S2+cal vs S2+cal +0.002 [+0.001, +0.004] (beats), vs SD +0.766 [+0.705, +0.826] (beats) → **complementary**

### Scale diagnostics: SD (full context)

| dataset | rounds | β̂ mean | β̂ max | at β̂ = 1 | at cap e^0.5 | N median | N p90 | N max |
|---|---|---|---|---|---|---|---|---|
| aime | 59351 | 1.193 | 1.424 | 19.1% | 0.0% | 5642 | 13539 | 16797 |
| lcb | 83959 | 1.209 | 1.438 | 13.9% | 0.0% | 6451 | 12959 | 18073 |
| all | 143310 | 1.202 | 1.438 | 16.1% | 0.0% | 6103 | 13216 | 18073 |

### Scale diagnostics: SD+S2+cal (S2 layout, LCB)

| dataset | rounds | β̂ mean | β̂ max | at β̂ = 1 | at cap e^0.5 | N median | N p90 | N max |
|---|---|---|---|---|---|---|---|---|
| lcb | 67727 | 1.014 | 1.093 | 42.9% | 0.0% | 2126 | 2426 | 3241 |
| all | 67727 | 1.014 | 1.093 | 42.9% | 0.0% | 2126 | 2426 | 3241 |
