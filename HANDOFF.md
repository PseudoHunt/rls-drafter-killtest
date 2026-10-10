# HANDOFF — RLS drafter kill test (paused 2026-10-10)

Read this first in a fresh session. Then read `KILL_TEST_BRIEF.md` (brief + amendments §11, §12), `NOTES.md` and
`RUNLOG.md`.

## Where things live (all under /home — the only thing that survives a JarvisLabs pause)

| what | path |
|---|---|
| repo / `$PERSIST` | `/home/rls_drafter_killtest` (git remote: `github.com/PseudoHunt/rls-drafter-killtest`, private) |
| model weights (`HF_HOME`) | `/home/rls_drafter_killtest/hf_cache` (Qwen/Qwen3-4B, z-lab/Qwen3-4B-DFlash-b16; not in git) |
| traces (not in git) | `/home/rls_drafter_killtest/traces/{traces,traces_nothink}.jsonl` |
| per-round replay logs | `/home/rls_drafter_killtest/results/raw/*.jsonl` (dev), `results/raw_nothink/` |
| venvs | `/home/envs/env-dflash` (torch 2.13, transformers 5.15, dflash 0.1.0), `/home/envs/env-vllm` (vLLM 0.31); interpreters in `/home/.local/share/uv/python` |
| resume | `source scripts/resume.sh` — sets env (offline HF), activates env-dflash, restarts MPS if a GPU is present, checks everything; prints `RESUME OK` |

The old copy at `/home/jl_fs/rls_drafter_killtest` (Ceph volume, 90% full) is stale; do not use it.

## Current state

- GPU used so far: 2.99 GPU-h (see RUNLOG.md). Stages 0–2 done; Stage 3 (dev) partially done, then stopped on request.
- Dev runs complete (12 dev traces, full traces; decision metric = flagged traces truncated per `results/loop_cuts.csv`):
  baseline; thinking-off control baseline; bias (hidden-residual) ρ ∈ {0.9, 0.99, 0.999}; RLS fp64, n0 ∈ {16, 128, 1024}
  × λ ∈ {0.995, 0.999, 0.9999, 1.0}. TTS: lr ∈ {1e-5, 3e-5, 1e-4} on 5 AIME dev traces; **TTS reference set**
  lr = 3e-5 on 4 dev traces (aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d). The TTS grid and the
  stale-vs-refresh check were cancelled (not needed after the RLS STOP).
- σ² = 8.628 (`configs/sigma2.json`). RLS runs in fp64 (fp32-vs-fp64 logit diff 3.2e-3 > 1e-3); λ = 1 recursion vs
  fixed solver in fp64: rel. err 1e-12 (`results/rls_checks.json`; fp32 round in `results/rls_checks_round1_fp32.json`).
- **RLS = STOP (futility rule §12.6).** Every bias and RLS config is *below* the baseline on dev (truncated, mean of
  AIME and LCB): baseline τ = 3.028; best RLS (n0 = 128, λ = 0.9999) τ = 2.739; best bias (ρ = 0.999) τ = 2.304. Dev R of
  the best RLS config is negative (< 0.15) → STOP; Stage 4 is not run. See `results/dev_interim.md` for the table.
- CPU diagnostic `scripts/adapt_diag.py` (does RLS cut out-of-sample hidden-state L2 error while lowering argmax
  accuracy?) — see "Diagnostic status" at the end of this file.

## Files that matter

- `rlsdk/replay.py` (round simulation, guard, TTS grad pass), `rlsdk/adapters.py` (Baseline, Bias, RLS, TTS),
  `rlsdk/run.py` (single-process runner), `rlsdk/pool.py` (concurrent MPS workers), `rlsdk/analyze.py` (pooled τ,
  truncation, bins), `rlsdk/data.py` (datasets, templates, loop flag/cut).
- `scripts/dev_table.py` (Gate C table), `scripts/summarize.py` (summary.md), `scripts/stage3.sh` (dev driver),
  `scripts/rls_checks.py`, `scripts/rvr_diag.py`, `scripts/adapt_diag.py`, `scripts/resume.sh`.
- `tests/` (40 CPU tests: `python -m pytest -q tests`, ~2 min).

## NEXT SESSION steps (copied in full from the user's instructions)

NEXT SESSION (CPU-only instance or my laptop; no GPU):
A. From the dev replay logs, compute per-slot conditional acceptance,
   P(a ≥ i | a ≥ i−1), for baseline, best RLS and best hidden-residual
   bias. Report slot 1, slots 2–4 and slots 5–15.
B. summary.md: RLS = STOP (futility rule §12.6). Include the dev table
   and the per-slot table. Label the bias arm "hidden-residual bias",
   not "Reflex-like".
C. Implement three arms, with CPU unit tests on the tiny model:
   1. KL-bias (Reflex form, no gate). Use g = Wᵀ(p − q) per valid slot
      (i ≤ a+1), weighted by w_i. Update m ← ρ·m + (1−ρ)·g/RMS(g) and
      apply z' = z + α·RMS(z)·m/RMS(m).
      Grid: ρ = 0.9, α ∈ {0.005, 0.01, 0.025}.
   2. KL-linear adapter: z' = z + A·z̃, A ∈ R^{d×(d+1)}, starting at 0.
      Loss: Σ_{i≤a+1} w_i · KL(p_i ‖ softmax(W z'_i)).
      Closed-form gradient, no autograd through the drafter:
      ∇A = Σ_i w_i · Wᵀ(q'_i − p_i) · z̃_iᵀ.
      Adam, lr ∈ {1e-7, 1e-6, 1e-5}. Keep A and Adam state in fp32. Log
      ‖A·z̃‖/‖z‖ per round and flag anything above 0.2.
   3. Slot-1-only RLS: fit and apply at slot 1 only, using the best
      (n0, λ) from the dev grid.
   Tests: closed-form gradient = autograd; A = 0 and α = 0 reproduce
   baseline bit-exactly; the leakage guard still raises; only pairs
   with i ≤ a+1 are used.
D. Write scripts/run_screen.sh. It should, in order:
   - run resume.sh;
   - run the 7 configs on the 4 traces, with MPS concurrency fitted to
     the card's memory (assume a 24 GB GPU);
   - write results/screen.md with τ, R = (τ − τ_base)/(τ_TTS − τ_base),
     update ms/round and state MB;
   - commit and push;
   - pause the instance via the JarvisLabs CLI/API (ask me for the API
     token). If that isn't possible, print a loud DONE so I can pause
     it by hand.
   Dry-run it end to end on CPU with the tiny model before any GPU
   resume.
E. Pre-register in RUNLOG.md before the GPU run:
   - best arm R ≥ 0.5: promising; we run the full held-out test later;
   - R < 0.25: stop this line;
   - in between: report and ask.
   Note it's tuned on the same 4 traces, so R is optimistic.

GPU SESSION: I'll resume on a cheap 24 GB GPU and run only
scripts/run_screen.sh.

## Notes for implementing the NEXT SESSION steps

- The 4 screen traces = the TTS reference set above (2 AIME + 2 LCB, none dropped, all non-looping):
  `aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d`. τ_base and τ_TTS for R come from the existing
  `results/raw/{trace}__baseline__default.jsonl` and `{trace}__tts__lr=3e-05.jsonl` on these 4 traces.
- 7 screen configs = KL-bias α ∈ {0.005, 0.01, 0.025} (ρ = 0.9) + KL-linear lr ∈ {1e-7, 1e-6, 1e-5} + slot-1 RLS.
- W = target lm_head weight (tied embeddings, [151936, 2560]); p_i = softmax(W h*_{t+i−1}), q_i = softmax(W z_i);
  w_i = exp(−(i−1)/7) (`rlsdk.adapters.dflash_position_weights`). Adapters plug into `simulate()` via
  `apply(z, slots)` / `update(z, h, slots, rnd)`; `rnd.target_logits` exists only for TTS today — compute W h* in
  the adapter (or extend replay to pass target logits for all methods).
- Memory on 24 GB: a light worker needs ≈ 3.8 GB, a teacher-forced pass ≈ 10.7 GB transient (serialised by the TF
  lock); KL arms need a few hundred MB more for the [m, V] fp32 softmaxes. 3 light workers fit.
- 24 GB cards may lack MPS permission; `pool.py` works without MPS (just less concurrency gain).
- Dry-run on CPU: `rlsdk.run --tiny --device cpu` exercises the same adapter paths (see `tests/test_resume.py`).

## Diagnostic status

Finished (`results/adapt_diag_aime24_I_13.json`, CPU, first 3,000 output tokens of aime24_I_13, along the baseline
trajectory; shadow adapters updated with valid pairs only and scored out-of-sample on all 15 slots of each next round):

| arm | hidden-state L2 error ‖h* − z′‖² | draft accuracy (all slots) | slot 1 accuracy |
|---|---|---|---|
| baseline | 27,314 | 31.7% | 76.4% |
| hidden-residual bias ρ = 0.99 | 17,644 | 26.6% | 71.8% |
| RLS n0 = 128, λ = 0.999 | 14,101 | 27.7% | 70.3% |
| RLS n0 = 1024, λ = 1 | 13,985 | 28.7% | 71.2% |

Target sanity: argmax(lm_head(h*)) = next token at 99.2% of positions. RLS works as a regressor (≈ 49% lower
out-of-sample L2) yet lowers accuracy at every slot: least squares on hidden states is not aligned with the drafter's
argmax decisions. This is why the KL-based arms in the next session are worth screening. It is a real effect, not a
bug.

## Reference numbers for the screen (4 traces, truncated variant; from `results/dev_interim.md`)

τ_base = 3.017, τ_TTS(lr = 3e-5) = 4.201. Best RLS on these 4 traces: τ = 2.724 (R = −0.25); best hidden-residual bias:
τ = 2.306 (R = −0.60).
