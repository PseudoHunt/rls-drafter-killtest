# Session 3, Step 5 — held-out (49 test traces, A30)

| arm | traces | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | τ 0–4K | update ms/round | rounds |
|---|---|---|---|---|---|---|---|---|---|
| S2 | 49 | 4.069 | 3.516 | 4.020 | 4.185 | 4.416 | 3.745 | 0.02 | 141040 |
| TTS stride 1 | 49 | 4.208 | 3.693 | 4.396 | 4.213 | 4.483 | 4.006 | 69.63 | 135679 |
| S2+cal | 49 | 4.408 | 3.779 | 4.370 | 4.578 | 4.756 | 4.046 | 0.02 | 130327 |
| full | 49 | 4.548 | 3.802 | 4.483 | 4.741 | 5.007 | 4.107 | 11.36 | 125881 |

| arm | arm − TTS stride 1 | arm − S2 | beats TTS? | update ms/round (ratio) | cheaper? |
|---|---|---|---|---|---|
| S2+cal | +0.200 [+0.179, +0.222] | +0.339 [+0.308, +0.371] | YES | 0.00 (0.0%) | YES |
| full | +0.340 [+0.319, +0.360] | +0.479 [+0.440, +0.520] | YES | 11.36 (16.3%) | no |

TTS stride 1 amortized update ms/round on these traces: 69.63.

## Per dataset (95% CI, bootstrap over that dataset's traces)

| arm | τ AIME | τ LCB |
|---|---|---|
| S2 | 4.276 [4.087, 4.471] (n=23) | 3.862 [3.755, 3.964] (n=26) |
| TTS stride 1 | 4.353 [4.151, 4.561] (n=23) | 4.063 [3.943, 4.188] (n=26) |
| S2+cal | 4.607 [4.384, 4.838] (n=23) | 4.209 [4.091, 4.330] (n=26) |
| full | 4.722 [4.497, 4.951] (n=23) | 4.374 [4.241, 4.510] (n=26) |

| contrast | AIME | LCB |
|---|---|---|
| S2+cal − TTS stride 1 | +0.254 [+0.225, +0.288] | +0.146 [+0.115, +0.172] |
| S2+cal − S2 | +0.331 [+0.288, +0.376] | +0.347 [+0.302, +0.395] |
| full − TTS stride 1 | +0.369 [+0.338, +0.399] | +0.311 [+0.282, +0.336] |
| full − S2 | +0.447 [+0.398, +0.497] | +0.512 [+0.451, +0.575] |

## Per bin vs TTS stride 1 (point estimates; CI in brackets)

| arm | 0–2K | 2–4K | 4–8K | ≥8K |
|---|---|---|---|---|
| S2+cal − TTS | +0.086 [+0.064, +0.109] | -0.026 [-0.068, +0.012]; closes 93% of S2's deficit (+0.376) | +0.364 [+0.324, +0.413] | +0.273 [+0.232, +0.332] |
| full − TTS | +0.109 [+0.089, +0.131] | +0.087 [+0.060, +0.115] | +0.528 [+0.484, +0.580] | +0.524 [+0.480, +0.584] |

Reference traces (used to pick the calibration lr) among these test traces: none (all 4 are dev traces; asserted in scripts/check_cal_disjoint.py).
