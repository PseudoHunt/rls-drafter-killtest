## Per-slot conditional acceptance P(a ≥ i | a ≥ i−1) — dev, flagged traces truncated

Counts pooled over rounds within a dataset; mean = average of AIME and LCB. A round counts toward slot i only if the (truncated) trace had a token at slot i.

| arm | slot 1 AIME | slot 1 LCB | slot 1 mean | slots 2–4 AIME | slots 2–4 LCB | slots 2–4 mean | slots 5–15 AIME | slots 5–15 LCB | slots 5–15 mean |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.723 | 0.701 | 0.712 | 0.677 | 0.612 | 0.645 | 0.690 | 0.604 | 0.647 |
| best RLS (n0=128, λ=0.9999) | 0.678 | 0.663 | 0.670 | 0.651 | 0.559 | 0.605 | 0.666 | 0.578 | 0.622 |
| best hidden-residual bias (ρ=0.999) | 0.623 | 0.584 | 0.604 | 0.567 | 0.484 | 0.525 | 0.612 | 0.519 | 0.566 |
