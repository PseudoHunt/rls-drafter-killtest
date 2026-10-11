# Session 4, item 2 — thinking-mode drafter (49 test traces, A30)

Drafter jgeuter/qwen3-4b-dpard-thinking-b16 @ 78baf299fc6dc793aab6c35ce44676a980e266d1 (3 layers, target feature layers [1, 17, 33]; released: 5 layers, [1, 9, 17, 25, 33]); no updates (0 ms/round).

| arm | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | update ms/round | draft ms/round* | rounds |
|---|---|---|---|---|---|---|---|---|
| TTS stride 1 | 4.208 | 3.693 | 4.396 | 4.213 | 4.483 | 69.63 | 7.4 | 135679 |
| S2+cal | 4.408 | 3.779 | 4.370 | 4.578 | 4.756 | 0.00 | 7.9 | 130327 |
| thinking baseline | 3.588 | 3.248 | 3.769 | 3.898 | 3.438 | 0.02 | 6.9 | 165845 |
| thinking + S2 | 3.790 | 3.247 | 3.759 | 3.917 | 4.109 | 0.02 | 6.7 | 151521 |

\* draft ms under 2 concurrent light workers for the new arms: indicative only.

## Contrasts (paired 95% CI)

| contrast | overall | 0–2K | 2–4K | 4–8K | ≥8K | AIME | LCB |
|---|---|---|---|---|---|---|---|
| thinking baseline − TTS stride 1 | -0.620 [-0.686, -0.552] | -0.445 [-0.505, -0.387] | -0.627 [-0.703, -0.555] | -0.315 [-0.393, -0.252] | -1.044 [-1.163, -0.916] | -0.760 [-0.857, -0.667] | -0.479 [-0.568, -0.379] |
| thinking baseline − S2+cal | -0.820 [-0.893, -0.742] | -0.531 [-0.595, -0.471] | -0.600 [-0.685, -0.520] | -0.679 [-0.796, -0.589] | -1.318 [-1.450, -1.186] | -1.014 [-1.126, -0.905] | -0.626 [-0.716, -0.516] |
| thinking + S2 − TTS stride 1 | -0.418 [-0.474, -0.366] | -0.446 [-0.508, -0.385] | -0.637 [-0.714, -0.566] | -0.296 [-0.374, -0.231] | -0.373 [-0.467, -0.295] | -0.557 [-0.659, -0.461] | -0.279 [-0.318, -0.242] |
| thinking + S2 − S2+cal | -0.618 [-0.686, -0.553] | -0.532 [-0.597, -0.469] | -0.610 [-0.695, -0.530] | -0.660 [-0.777, -0.569] | -0.646 [-0.772, -0.550] | -0.811 [-0.939, -0.692] | -0.425 [-0.471, -0.376] |

## Per dataset (95% CI, bootstrap within the dataset)

| arm | τ AIME | τ LCB |
|---|---|---|
| TTS stride 1 | 4.353 [4.151, 4.561] (n=23) | 4.063 [3.943, 4.188] (n=26) |
| S2+cal | 4.607 [4.384, 4.838] (n=23) | 4.209 [4.091, 4.330] (n=26) |
| thinking baseline | 3.593 [3.451, 3.749] (n=23) | 3.583 [3.473, 3.716] (n=26) |
| thinking + S2 | 3.797 [3.664, 3.927] (n=23) | 3.784 [3.668, 3.906] (n=26) |

## Further contrasts (paired 95% CI)

| contrast | CI |
|---|---|
| thinking + S2 − thinking baseline | +0.202 [+0.141, +0.255] |
| thinking baseline − baseline | +0.488 [+0.395, +0.561] |
| thinking + S2 − S2 | -0.278 [-0.341, -0.221] |

## Reading (pre-registered)

- thinking baseline − S2+cal -0.820 [-0.893, -0.742] (below) → **thinking baseline below S2+cal.**
