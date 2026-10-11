# Session 4, item 3 — cal-full (A30)

cal-full = released drafter calibrated in the full-context layout (same data, anchors, steps, lr as S2+cal), replayed with full context (no window). 0 ms/round.

| arm | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | update ms/round | draft ms/round* | rounds |
|---|---|---|---|---|---|---|---|---|
| TTS stride 1 | 4.208 | 3.693 | 4.396 | 4.213 | 4.483 | 69.63 | 7.4 | 135679 |
| S2+cal | 4.408 | 3.779 | 4.370 | 4.578 | 4.756 | 0.00 | 7.9 | 130327 |
| cal-full | 4.166 | 3.763 | 4.322 | 4.342 | 4.187 | 0.02 | 9.5 | 138961 |

\* draft ms under 2 concurrent light workers for the new arms: indicative only.

## Contrasts (paired 95% CI)

| contrast | overall | 0–2K | 2–4K | 4–8K | ≥8K | AIME | LCB |
|---|---|---|---|---|---|---|---|
| cal-full − TTS stride 1 | -0.042 [-0.077, -0.001] | +0.070 [+0.046, +0.094] | -0.074 [-0.117, -0.035] | +0.129 [+0.095, +0.171] | -0.295 [-0.346, -0.234] | +0.011 [-0.040, +0.077] | -0.095 [-0.143, -0.048] |
| cal-full − S2+cal | -0.242 [-0.272, -0.206] | -0.016 [-0.030, -0.004] | -0.047 [-0.066, -0.029] | -0.236 [-0.267, -0.207] | -0.568 [-0.621, -0.515] | -0.243 [-0.286, -0.188] | -0.241 [-0.285, -0.196] |

## Per dataset (95% CI, bootstrap within the dataset)

| arm | τ AIME | τ LCB |
|---|---|---|
| TTS stride 1 | 4.353 [4.151, 4.561] (n=23) | 4.063 [3.943, 4.188] (n=26) |
| S2+cal | 4.607 [4.384, 4.838] (n=23) | 4.209 [4.091, 4.330] (n=26) |
| cal-full | 4.365 [4.143, 4.605] (n=23) | 3.968 [3.875, 4.071] (n=26) |

## Further contrasts (paired 95% CI)

| contrast | CI |
|---|---|
| cal-full − baseline | +1.066 [+0.976, +1.141] |
| cal-full − S2 | +0.097 [+0.058, +0.144] |
| S2+cal − cal-full | +0.242 [+0.206, +0.272] |
| S2+cal − cal-full, 0–2K | +0.016 [+0.004, +0.030] |
| S2+cal − cal-full, 2–4K | +0.047 [+0.029, +0.066] |
| S2+cal − cal-full, 4–8K | +0.236 [+0.207, +0.267] |
| S2+cal − cal-full, ≥8K | +0.568 [+0.515, +0.621] |

## Reading (pre-registered)

- cal-full − S2+cal -0.242 [-0.272, -0.206] (below).
