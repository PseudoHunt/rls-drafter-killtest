# Step 1 — exact batched pooled update (16 T2 traces, A30)

Pre-registered in RUNLOG.md (session 3). Decision variant; one TTS process alone on the GPU for every row.

| arm | traces | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | τ 0–4K | update ms/round | rounds |
|---|---|---|---|---|---|---|---|---|---|
| TTS stride 1 (existing) | 16 | 4.299 | 3.631 | 4.448 | 4.386 | 4.725 | 3.983 | 69.60 | 40747 |
| hybrid, per-round pooled (existing) | 16 | 4.429 | 3.513 | 4.237 | 4.771 | 5.191 | 3.827 | 43.01 | 39512 |
| batched pooled stride 10 | 16 | 4.426 | 3.511 | 4.242 | 4.764 | 5.181 | 3.828 | 11.49 | 39546 |

| variant | amortized update ms/round | ratio to TTS stride 1 | ≤ 10%? | draft ms/round |
|---|---|---|---|---|
| TTS stride 1 (existing) | 69.60 | 100.0% | no | 7.35 |
| hybrid, per-round pooled (existing) | 43.01 | 61.8% | no | 8.18 |
| batched pooled stride 10 | 11.49 | 16.5% | no | 8.11 |

- **τ check:** batched stride 10 τ = 4.426 vs per-round pooled hybrid 4.429 (|Δ| = 0.003 vs 4.429; need ≤ 0.02) → **PASS**
- batched stride 10 − TTS stride 1: +0.127 [+0.060, +0.173]
- cost criterion (≤ 10% of TTS stride 1 = 6.96 ms): batched stride 10 11.49 ms → **not met**

| trace | hybrid τ | batched s10 τ | Δ |
|---|---|---|---|
| aime24_II_01 | 4.916 | 4.916 | +0.000 |
| aime24_II_10 | 4.343 | 4.341 | -0.003 |
| aime24_II_12 | 4.385 | 4.380 | -0.005 |
| aime24_I_01 | 4.474 | 4.488 | +0.014 |
| aime24_I_03 | 4.499 | 4.501 | +0.002 |
| aime24_I_04 | 4.203 | 4.189 | -0.014 |
| aime24_I_05 | 4.665 | 4.647 | -0.018 |
| aime24_I_14 | 5.123 | 5.113 | -0.010 |
| lcb_3697 | 4.754 | 4.758 | +0.004 |
| lcb_abc391_e | 3.885 | 3.877 | -0.008 |
| lcb_abc391_g | 4.351 | 4.339 | -0.013 |
| lcb_abc396_e | 3.678 | 3.682 | +0.003 |
| lcb_arc190_c | 4.181 | 4.192 | +0.011 |
| lcb_arc192_b | 3.529 | 3.525 | -0.005 |
| lcb_arc192_e | 4.993 | 4.970 | -0.022 |
| lcb_arc194_d | 4.041 | 4.030 | -0.011 |

Stride 20 (exploratory) ran on 15/16 traces only (budget): aime24_II_01, aime24_II_10, aime24_II_12, aime24_I_01, aime24_I_03, aime24_I_04, aime24_I_05, aime24_I_14, lcb_3697, lcb_abc391_e, lcb_abc391_g, lcb_abc396_e, lcb_arc190_c, lcb_arc192_b, lcb_arc192_e; table on that subset below.

| arm | traces | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | τ 0–4K | update ms/round | rounds |
|---|---|---|---|---|---|---|---|---|---|
| TTS stride 1 (existing) | 15 | 4.299 | 3.629 | 4.402 | 4.380 | 4.725 | 3.963 | 69.73 | 38458 |
| hybrid, per-round pooled (existing) | 15 | 4.438 | 3.511 | 4.196 | 4.772 | 5.191 | 3.810 | 43.03 | 37235 |
| batched pooled stride 10 | 15 | 4.435 | 3.508 | 4.203 | 4.767 | 5.181 | 3.810 | 11.49 | 37261 |
| batched pooled stride 20 (exploratory) | 15 | 4.355 | 3.495 | 4.089 | 4.656 | 5.080 | 3.757 | 16.34 | 37958 |
