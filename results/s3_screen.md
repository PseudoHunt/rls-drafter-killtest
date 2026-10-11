# Session 3, Step 4 — screen (4 reference traces, A30)

cal checkpoint: `results/raw_a30_cal_lr1e-05`

| arm | traces | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | τ 0–4K | update ms/round | rounds |
|---|---|---|---|---|---|---|---|---|---|
| S2 | 4 | 4.231 | 3.256 | 4.013 | 4.251 | 4.779 | 3.594 | 0.02 | 12609 |
| TTS stride 1 | 4 | 4.319 | 3.462 | 4.413 | 4.221 | 4.719 | 3.880 | 70.50 | 12366 |
| S2+cal (static) | 4 | 4.540 | 3.488 | 4.395 | 4.599 | 5.044 | 3.889 | 0.02 | 11749 |
| full = S2+cal + batched TTS s10 | 4 | 4.678 | 3.522 | 4.496 | 4.760 | 5.264 | 3.949 | 10.98 | 11410 |

| trace | S2 | TTS stride 1 | S2+cal (static) | full = S2+cal + batched TTS s10 |
|---|---|---|---|---|
| aime24_II_14 | 4.926 | 5.024 | 5.316 | 5.432 |
| aime24_I_10 | 4.198 | 4.200 | 4.487 | 4.607 |
| lcb_abc393_d | 3.742 | 3.816 | 4.004 | 4.066 |
| lcb_arc191_d | 4.059 | 4.236 | 4.360 | 4.587 |

Readings (pre-registered):
- Mode mismatch: S2 0–4K τ 3.594, TTS 3.880 (deficit +0.286), S2+cal 3.889 → closes 103% (need ≥ 50%) → **supported**
- Promote full: τ_full 4.678 vs τ_TTS + 0.10 = 4.419; full > TTS on 4/4 traces → **PROMOTE**
- Promote S2+cal: τ_S2cal 4.540 vs τ_TTS 4.319 → **PROMOTE**
- Selection optimism: the S2+cal lr was chosen on these 4 traces (TTS lr also dev-tuned).
