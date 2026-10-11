# Session 4, item 1d — SharpDraft sanity anchor (4 reference traces, released drafter, full context, A30)

| arm | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K |
|---|---|---|---|---|---|
| baseline | 3.016 | 3.271 | 3.917 | 2.996 | 2.655 |
| SharpDraft | 3.711 | 3.271 | 3.995 | 3.613 | 3.801 |
| Δ SharpDraft − baseline | +0.695 | +0.000 | +0.078 | +0.617 | +1.146 |

| trace | baseline | SharpDraft |
|---|---|---|
| aime24_II_14 | 3.253 | 4.305 |
| aime24_I_10 | 2.831 | 3.621 |
| lcb_abc393_d | 3.070 | 3.444 |
| lcb_arc191_d | 2.964 | 3.519 |

### Scale diagnostics: SharpDraft, full context

| dataset | rounds | β̂ mean | β̂ max | at β̂ = 1 | at cap e^0.5 | N median | N p90 | N max |
|---|---|---|---|---|---|---|---|---|
| aime | 8331 | 1.235 | 1.420 | 14.4% | 0.0% | 7827 | 14389 | 16533 |
| lcb | 6059 | 1.194 | 1.382 | 14.6% | 0.0% | 6060 | 11586 | 13617 |
| all | 14390 | 1.218 | 1.420 | 14.5% | 0.0% | 6891 | 13189 | 16533 |

Rule (pre-registered): Δ overall +0.695 > 0, Δ(4–8K) +0.617 > Δ(0–2K) +0.000, Δ(≥8K) +1.146 > Δ(0–2K) → **PASS**
