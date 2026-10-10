# Proposal (not run): real, non-replay loop check — baseline, window, S2 on 3 dev traces

Why: every τ so far is from offline replay against a fixed greedy trace (Stage 2 checked replay vs the real loop
for the baseline only: τ within 0.41%, round-identity 89.6–95.3% because of bf16 feature numerics). The window and S2
change only the drafter's context, so greedy outputs must stay identical; this check confirms that, measures τ in the
real loop, and gives real end-to-end tokens/s.

## Design
- Traces: 3 dev traces, not in the 4-trace screen set: aime24_II_09, aime24_II_13, lcb_3701 (one long, one medium,
  one LCB). Same prompts, greedy, max 16,384 new tokens, thinking on (as the traces).
- Arms: baseline DFlash; window (W 1024 + prompt); S2 (W 1024 + prompt + 16 far chunks, target layers 17/25/33 scored
  at t'−1). Implemented inside `dflash.model.dflash_generate`'s loop:
  - window / S2: the drafter's context cache becomes `rlsdk.window.WindowCache` (pre-RoPE keys, contiguous
    positions);
  - S2: the target's verification forward also returns the attention row at the last committed position for layers
    17/25/33 (one query row against the resident K cache; head-summed, 32-token chunks).
- Checks:
  1. **Output identity:** each arm's generated tokens equal the baseline's (greedy target; drafts only change
     acceptance). Any mismatch = a bug in the integration.
  2. **τ, real vs replay:** real-loop τ per arm vs the replay τ on the same trace, within 2%.
  3. **Speed:** wall-clock tokens/s and per-round drafting / target ms, one process alone on the GPU, 2 repeats.
- Cost: 3 traces × 3 arms × ≈ 2–4 min ≈ 0.5 A30-h, plus ≈ 1 h of integration coding (CPU tests on the tiny model
  first: real loop with window/S2 at W ≥ context = baseline bit-exact).
- Decision: report only (sanity gate before any deployment claim). If output identity fails, the held-out
  conclusions stand (they are replay-exact by construction) but the integration must be fixed before speed claims.
