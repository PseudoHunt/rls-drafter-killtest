# T0b: target-guided far-context selection (4 TTS-reference traces, A30)

τ_base = 3.016; window (frozen) = 4.085; gain_TTS = 1.184 (old GPU); R = (τ − τ_base)/gain_TTS. Far chunks = aligned 32-token chunks wholly older than the window (and after the prompt), re-selected every round from the target's head-summed attention row at the last committed position t'−1 (guarded), kept in original order, renumbered contiguously. Draft ms/round under 2 concurrent workers (MPS), indicative; baseline and window rows from the earlier window run (3 workers).

| config | τ | R | Δτ vs window | τ AIME | τ LCB | τ <2K | τ ≥8K | draft ms/round | mean visible keys |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3.016 | +0.000 | -1.069 | 3.028 | 3.004 | 3.271 | 2.655 | 12.00 | nan |
| window W 1024 + prompt (frozen) | 4.085 | +0.903 | +0.000 | 4.348 | 3.822 | 3.246 | 4.535 | 11.58 | 1310 |
| S1: W 1024 + prompt + 16 far chunks, target last layer | 4.192 | +0.993 | +0.107 | 4.476 | 3.908 | 3.255 | 4.696 | 9.20 | 1754 |
| S2: W 1024 + prompt + 16 far chunks, target layers 17/25/33 | 4.231 | +1.026 | +0.146 | 4.533 | 3.929 | 3.256 | 4.779 | 8.38 | 1754 |
| S3: W 512 + prompt + 16 far chunks, last layer (window budget) | 4.157 | +0.963 | +0.072 | 4.438 | 3.875 | 3.249 | 4.618 | 8.25 | 1308 |
| C1: window W 1536 + prompt (S1 budget) | 4.124 | +0.936 | +0.039 | 4.385 | 3.863 | 3.264 | 4.587 | 8.33 | 1757 |
| C2: W 1024 + prompt + 16 random far chunks (seed 0) | 4.134 | +0.944 | +0.049 | 4.407 | 3.860 | 3.264 | 4.594 | 8.80 | 1756 |

## Pre-registered rule (RUNLOG.md)

Best of S1/S2/S3: **S2**, τ = 4.231: vs window +0.146 (need ≥ +0.1), vs C1 +0.107, vs C2 +0.097 (need > 0 for both).

**Target-guided selection adds value → freeze S2 and add it as an arm in T1 and T2.**

## Extra target-side cost of the selection signal (A30 microbenchmark, scripts/attn_row_cost.py)

One attention row at t'−1 per round from the resident KV cache, head-summed, chunk-reduced, top-16. Unfused PyTorch (fp32 copy of K), so an upper bound; the replay itself precomputes all rows in the teacher-forced pass, which is not a deployment cost.

| context T | 1 layer (S1/S3) ms | 3 layers (S2) ms |
|---|---|---|
| 2048 | 0.223 | 0.563 |
| 8192 | 0.337 | 0.657 |
| 16384 | 0.406 | 1.122 |
