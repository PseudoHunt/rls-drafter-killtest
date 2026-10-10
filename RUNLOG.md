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

| # | start (UTC) | end (UTC) | command | A30-h | cum A30-h | note |
|---|---|---|---|---|---|---|
| 16 | 2026-10-10 09:15:38 | 09:21:16 | instance idle (awaiting instructions) | 0.094 | 0.662 | counted: instance was running |
| 17 | 2026-10-10 09:21:16 | 09:30:36 | coding on the instance: KL STOP recorded, NOTES §9 (DFlash RoPE), rlsdk/window.py + 9 CPU tests (63/63 pass), GPU check W = 1e6 on lcb_abc393_d (bit-identical to baseline), tiny dry-runs, pre-registration | 0.156 | 0.817 | smoke test W = 1024 keep 64 on lcb_abc393_d: τ = 3.588 (seen before pre-registering; disclosed there) |
| 18 | 2026-10-10 09:30:36 | 09:36:40 | `SCREEN=window WORKERS=3 bash scripts/run_screen.sh` (baseline + 4 window configs × 4 traces) | 0.101 | 0.919 | best = W 1024, keep prompt: τ = 4.085, R = +0.903 → **promising**; τ ≥8K 2.655 → 4.535, τ <2K unchanged |
| 19 | 2026-10-10 09:36:40 | 09:37:02 | wrap-up: summary.md, RUNLOG, commit/push | 0.006 | 0.925 | instance still running; pause by hand |

**A30 session total (instance time incl. coding and idle, 08:41:09–09:37:02 UTC): 0.931 A30-h.** Old-GPU total stays 2.991 GPU-h.

**Status:** KL screen = STOP. Windowed-context test = **promising** (best R = +0.90, pre-registered rule): per that rule, the next step is a full held-out test (not run). See results/window.md.

**Pre-registered before the mechanism + selection screen (2026-10-10 09:54 UTC):**
- Same 4 reference traces; A30 baseline τ = 3.016, gain_TTS = 1.184; target, verification and eval unchanged.
  Frozen window = W = 1024, keep prompt (τ = 4.085). Configs (NOTES.md §10), key budget 1024 where keys are selected:
  B window with original position ids; E all keys, distance capped at 4096; C exact top-1024 per head per query,
  original positions; C′ top-1024 under E's capped rotation; D as C′ but one top-1024 set per head shared over the
  16 block queries (summed logits).
- Readings: E within 0.1 τ of the window → distance is the main cause. E near the baseline while the window is far
  above → key count or distant noise is the cause. C′ or D beats the window by ≥ 0.1 τ → selection adds value,
  CrossCov-U goes next.
- Reported per config: τ, R, τ <2K, τ ≥8K, τ AIME, τ LCB, drafting ms/round.
- After the screen: STOP at a gate and propose the held-out test (baseline, frozen window, + best of C′/D/E only if
  it beats the window by ≥ 0.1 τ here; TTS lr = 3e-5 on all test traces; window + TTS on 16 test traces). Test
  decision rule: window R ≥ 0.5 confirms the headline. Not launched without approval.
- Checks before the run: 18 CPU tests (tests/test_ctxpolicy.py; no-op settings bit-exact vs baseline, active B/E =
  fresh forward, top-k mask and masked softmax vs references); 81/81 tests pass. GPU: E with cap = 1e6 on
  lcb_abc393_d = baseline, 2,675/2,675 rounds. Before pre-registering I also saw C′ (k = 1024, cap = 4096) on
  lcb_abc393_d alone: τ = 3.514 (timing probe; part of the grid, re-run there).

| # | start (UTC) | end (UTC) | command | A30-h | cum A30-h | note |
|---|---|---|---|---|---|---|
| 20 | 2026-10-10 09:37:02 | 09:46:17 | instance idle (awaiting instructions) | 0.154 | 1.079 | counted: instance was running |
| 21 | 2026-10-10 09:46:17 | 09:54:29 | coding on the instance: context policies B/E/C/C′/D (window.py generalised), 18 CPU tests (81/81 pass), mechanism report, GPU checks (E cap = 1e6 bit-identical to baseline; C′ timing probe on lcb_abc393_d), tiny dry-run, pre-registration | 0.137 | 1.216 | C′ on lcb_abc393_d τ = 3.514 seen before pre-registering (disclosed there) |
| 22 | 2026-10-10 09:54:29 | 09:59:00 | `SCREEN=mech WORKERS=3 bash scripts/run_screen.sh` — first attempt | 0.075 | 1.291 | **CUDA OOM** in one worker (TF pass while a top-k worker held 12 GB: allocator fragmentation from per-round variable-size tensors); stopped; 6 finished jobs kept |
| 23 | 2026-10-10 09:59:00 | 10:08:15 | `SCREEN=mech WORKERS=2 bash scripts/run_screen.sh` (relaunch, expandable_segments) | 0.154 | 1.445 | best = window (frozen) τ 4.085; B 4.001, D 3.899, C′ 3.839, E 3.539 (49% of window gain), C 3.198 → no selection gain; no extra held-out arm |
| 24 | 2026-10-10 10:08:15 | 10:10:31 | report fixes (E reading, timing note), TTS cost probe on A30 (400 rounds, 62 ms/round, peak 13.4 GB), held-out proposal, summary/RUNLOG, commit/push | 0.038 | 1.483 | GATE: held-out test proposed (results/heldout_proposal.md), not launched |

**A30 session total (instance time incl. coding and idle, 08:41:09–10:10:31 UTC): 1.489 A30-h.** Old-GPU total stays 2.991 GPU-h.

**Status:** mechanism screen done: the frozen window (W = 1024, keep prompt) stays best; selection (C′/D) does not beat it; E recovers 49% of its gain. **GATE** — held-out test proposed in results/heldout_proposal.md (projection ≈ 5–6 A30-h, range 4.5–7); waiting for approval.

**Pre-registered before T0b, target-guided selection screen (2026-10-10 10:31 UTC; after T0 = mechanism screen, before T1):**
- Same 4 reference traces; A30 baseline τ = 3.016; frozen window (W 1024, keep prompt) τ = 4.085; gain_TTS 1.184.
  Target, verification and eval unchanged. NOTES.md §11.
- Signal: target's head-summed attention row at the last committed position t'−1 (guard allows only indices ≤ t'−1),
  scored over 32-token chunks older than the window, top chunks kept, refreshed every round, kept keys renumbered
  contiguously in original order.
- Configs: S1 W 1024 + prompt + 16 far chunks (target last layer, 35); S2 same, mean of layers 17/25/33; S3 W 512 +
  prompt + 16 far chunks (S1 scoring; window budget); C1 window W 1536 + prompt (S1 budget); C2 W 1024 + prompt + 16
  random far chunks (seed 0, re-drawn each round).
- **Rule:** target-guided selection adds value only if the best of S1/S2/S3 beats the window by ≥ 0.1 τ **and** beats
  both C1 and C2. If so: freeze it and add it as an arm in T1 and T2 alongside the window. If not: plan unchanged.
- Reported: τ, τ AIME, τ LCB, τ ≥ 8K (and < 2K), draft ms/round; extra target-side cost of the attention row per
  round (A30 microbenchmark, results/attn_row_cost.md: 0.22–0.41 ms for 1 layer, 0.56–1.12 ms for 3 layers at
  2K–16K context).
- Checks before the run: 11 CPU tests (tests/test_farsel.py: chunk scores = eager attention; no far context → S1 =
  window bit-exact; active selection = fresh forward on the visible rows, chunks = top-n of row t'−1; guard reads
  only row t'−1); 92/92 tests pass. GPU on lcb_abc393_d: S1 with W = 1e6 = baseline (2,675/2,675 rounds). Seen
  before pre-registering: S1 on lcb_abc393_d alone τ = 3.716 (window there 3.653); it is re-run in the grid.

| # | start (UTC) | end (UTC) | command | A30-h | cum A30-h | note |
|---|---|---|---|---|---|---|
| 25 | 2026-10-10 10:10:31 | 10:22:56 | instance idle (awaiting instructions) | 0.207 | 1.690 | counted: instance was running |
| 26 | 2026-10-10 10:22:56 | 10:31:24 | coding on the instance: T0b target attention summaries (rlsdk/target_attn.py), far-chunk selection in WindowCache, FarSel arm, 11 CPU tests (92/92 pass), GPU check (S1 W = 1e6 = baseline), attention-row cost microbenchmark, tiny dry-run, pre-registration | 0.141 | 1.831 | S1 on lcb_abc393_d τ = 3.716 seen before pre-registering (disclosed there) |
| 27 | 2026-10-10 10:31:24 | 10:37:18 | `SCREEN=farsel WORKERS=2 bash scripts/run_screen.sh` (S1–S3, C1, C2 × 4 traces) | 0.098 | 1.929 | **rule met**: S2 τ 4.231, +0.146 vs window, +0.107 vs C1, +0.097 vs C2; positive on all 4 traces → S2 frozen for T1/T2 |
| 28 | 2026-10-10 10:37:18 | 10:38:09 | write-up: summary.md, held-out proposal (S2 arm, projection 6–7 A30-h), RUNLOG, commit/push | 0.014 | 1.944 | GATE unchanged: T1/T2 await approval |

**A30 session total (instance time incl. coding and idle, 08:41:09–10:38:09 UTC): 1.950 A30-h.** Old-GPU total stays 2.991 GPU-h.

**Status:** T0b = target-guided selection adds value under the pre-registered rule → **S2 frozen** (W 1024 + prompt + 16 far chunks scored by target layers 17/25/33) and added to T1 and T2 alongside the window. GATE — held-out plan updated in results/heldout_proposal.md (≈ 6–7 A30-h, range 5.5–8); waiting for approval.
