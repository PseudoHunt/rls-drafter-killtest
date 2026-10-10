# RUNLOG

Venvs live on local disk at `/home/envs/{env-dflash,env-vllm}` (rebuild: `uv venv -p 3.12`, then
`uv pip install "dflash[local]==0.1.0" pytest pyyaml matplotlib numpy pandas accelerate` /
`uv pip install vllm datasets pyyaml`; exact pins in results/ENV.md).
`PERSIST=/home/jl_fs/rls_drafter_killtest` (moved to `/home/rls_drafter_killtest` on 2026-10-10 for pause/resume), `HF_HOME=$PERSIST/hf_cache`.

| # | start (UTC) | end (UTC) | command | GPU-h | cum GPU-h | note |
|---|---|---|---|---|---|---|
| 0 | 2026-10-10 03:55 | 03:58 | env setup (uv venvs) | 0 | 0 | CPU |
| 1 | 2026-10-10 04:14 | 04:16 | `pytest -q tests` (CPU, CUDA_VISIBLE_DEVICES="") | 0 | 0 | 28 passed |
| 2 | 2026-10-10 04:31:36 | 05:00:06 | `tmux: bash scripts/stage1.sh` (vLLM 0.31, 62 thinking-ON traces + 6 thinking-OFF dev-AIME control) | 0.475 | 0.475 | 62 traces 1192 s + control 196 s + 2×model load; 17/62 flagged looping → §9 stop |
| 3 | 2026-10-10 05:18:33 | 05:36:14 | `tmux: TIMING_ID=lcb_3701 bash scripts/stage2.sh` (real-vs-replay 3 dev prompts @4K + timing probe, 5 configs on lcb_3701) | 0.295 | 0.770 | RVR: τ diff 0.41% ✓, rounds identical by t 89.6–95.3% ✗ (<98%) |
| 4 | 2026-10-10 05:36:26 | 05:37:43 | `scripts/rvr_diag.py --trace-id aime24_II_09` | 0.021 | 0.791 | replay with recorded real features: 100% identical → mismatch = bf16 feature numerics |
| 5 | 2026-10-10 05:57:21 | 06:01:59 | `tmux: bash scripts/bench_mps.sh` (alone vs 6 workers under MPS, 1500 rounds) | 0.077 | 0.868 | RLS logs identical; TTS differed (round 41) → nondeterministic SDPA backward; 1 worker OOM (6 workers too many) |
| 6 | 2026-10-10 06:04:40 | 06:07:33 | `bash scripts/bench_det.sh` (TTS math-SDPA + deterministic algos: alone×2, MPS×2, 600 rounds) | 0.048 | 0.916 | all 4 identical; 2 concurrent TTS: 1.2× slower each |
| 7 | 2026-10-10 06:08:54 | 06:18:19 | `tmux: bash scripts/stage3.sh` (dev baseline ×12 + nothink ×6 under MPS; σ²; RLS checks) | 0.157 | 1.073 | σ²=8.628; fp32-vs-fp64 logits 3.2e-3 > 1e-3 → RLS fp64; λ=1 check in fp32 rel_M 1.8e-3 → driver aborted |
| 8 | 2026-10-10 06:19:26 | 07:33:39 | `tmux: bash scripts/stage3.sh` relaunch (baseline/nothink skipped; RLS checks fp64; bias+RLS grid ×12 dev, 5 workers MPS) | 1.237 | 2.310 | RLS fp64; λ=1 rel_M 1.0e-12 ✓; all bias/RLS < baseline on dev |
| 9 | 2026-10-10 07:33:39 | 08:10:30 | same driver, TTS grid (2 heavy workers) — cancelled on request (pause workflow) | 0.614 | 2.924 | completed: lr∈{1e-5,3e-5,1e-4} on 5 AIME dev traces |
| 10 | 2026-10-10 08:11:41 | 08:15:43 | `rlsdk.pool` TTS lr=3e-5 on lcb_abc393_d, lcb_arc191_d (MPS, 2 workers) | 0.067 | 2.991 | TTS reference set complete (4 traces) |
| 11 | 2026-10-10 07:39 | 08:15 | `scripts/adapt_diag.py` (CPU only, aime24_I_13, 3K tokens) | 0 | 2.991 | RLS cuts out-of-sample hidden L2 by ~49% but lowers draft accuracy 31.7%→28.7% |

**Status at pause:** RLS = STOP (futility rule §12.6, dev R < 0). Stage 4 not run. Next steps: HANDOFF.md.

## A30 session (2026-10-10, NVIDIA A30 24 GB; no pause between coding and screen)

**Pre-registered before the screen run (HANDOFF step E, as amended for the A30):**
- Screen = 6 configs on the 4 TTS-reference traces (aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d):
  KL-bias ρ = 0.9, α ∈ {0.005, 0.01, 0.025}; KL-linear Adam lr ∈ {1e-7, 1e-6, 1e-5}. Slot-1-only RLS dropped. All
  adaptation math in fp32.
- τ_base is re-measured on the A30 (flag if it differs from 3.017 by > 1%). gain_TTS is kept from the old GPU:
  4.201 − 3.017 = 1.184. R = (τ_arm − τ_base,A30) / 1.184.
- A30 baseline (row 12): τ = 3.016 (−0.04% vs 3.017, within 1%) → thresholds τ = 3.608 (R = 0.5), 3.312 (R = 0.25).
- Decision on the best arm: R ≥ 0.5 → promising, run the full held-out test later; R < 0.25 → stop this line;
  in between → report and ask.
- The configs are selected on the same 4 traces that measure R, so the best arm's R is optimistic.

| # | start (UTC) | end (UTC) | command | A30-h | cum A30-h | note |
|---|---|---|---|---|---|---|
| 12 | 2026-10-10 08:41:34 | 08:44:26 | `rlsdk.pool` baseline on the 4 reference traces → `results/raw_a30` (3 workers, MPS) | 0.048 | 0.048 | τ = 3.016 (old GPU 3.017); per-round trajectories diverge early (bf16 numerics, as in Stage 2) |
| 13 | 2026-10-10 08:44:26 | 08:54:10 | coding on the instance: KL arms + 14 CPU tests (54/54 pass), slot_accept.py, summary.md, run_screen.sh, real-model memory probe (400 rounds), tiny dry-runs ×2, stale-copy check + delete | 0.162 | 0.210 | probe: 1 worker peaks 13.4 GB (TF pass), ~5.5 GB steady → 2 workers |
| 14 | 2026-10-10 08:54:10 | 09:15:25 | `bash scripts/run_screen.sh` (6 configs × 4 traces, 2 workers, MPS) | 0.354 | 0.564 | best = KL-linear lr=1e-6 τ = 3.162, R = +0.123 → **stop this line** |
| 15 | 2026-10-10 09:15:25 | 09:15:38 | wrap-up: RUNLOG, commit/push | 0.004 | 0.568 | instance still running; pause by hand |

**A30 session total (instance time incl. coding, 08:41:09–09:15:38 UTC): 0.575 A30-h** (the 25 s before row 12 was resume.sh). Old-GPU total stays 2.991 GPU-h.

**Status:** screen decision = **stop this line** (pre-registered rule; best R = +0.123 < 0.25, optimistic since tuned on the same 4 traces). KL-linear lr = 1e-5 crossed ‖Az̃‖/‖z‖ > 0.2 in 34 rounds (max 0.209). See results/screen.md.

**KL screen = STOP** (recorded 2026-10-10 09:25): best R = +0.12 (KL-linear lr = 1e-6) < 0.25. Measured cost: 7–9 ms
per round of update and a 1.45 GiB fp32 lm_head copy (+75 MB KL-linear state).

**Pre-registered before the windowed-context run (2026-10-10 09:30 UTC):**
- Question: is the long-output decay mostly a long-context effect inside the drafter? Training-free: the drafter's
  context attention is limited to a kept set + the last W committed positions, contiguous position ids inside the
  kept cache (StreamingLLM-style; NOTES.md §9). Target and verification unchanged.
- Grid (4 configs): W ∈ {1024, 4096} × kept set ∈ {first 64 tokens, full prompt}; same 4 reference traces.
  Baseline re-run in the same pool (bit-identical τ, comparable drafting time).
- R = (τ − 3.016) / 1.184 (A30 baseline, old-GPU TTS gain). Best R ≥ 0.5 → promising, plan a full held-out test;
  R < 0.25 → close the long-output decay line; in between → report and ask. Also reported: τ for output positions
  < 2K and ≥ 8K, and drafting ms/round. Configs chosen on the same 4 traces → best R optimistic.
- Checks done before the run: 9 CPU unit tests (`tests/test_window.py`; bit-exact vs baseline when the context
  fits, equals a fresh forward on the visible rows otherwise); real model on lcb_abc393_d with W = 1e6: 2,675/2,675
  rounds identical to the A30 baseline. One look at the data before pre-registering: W = 1024, keep 64 on
  lcb_abc393_d alone gave τ = 3.588 (a smoke test of the code path; it is part of the grid and is re-run there).
