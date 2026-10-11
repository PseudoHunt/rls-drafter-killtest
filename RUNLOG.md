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

**Pre-registered before any held-out run (2026-10-10 11:11 UTC): S2 dominance and the S2 + TTS hybrid (P3 arms).**
- New arms on the same 16 test traces as S2 + TTS (8 AIME + 8 LCB, fixed seed, listed before the run):
  (a) **TTS stride 10** (TTS paper protocol): one Adam step every 10 rounds with valid pairs, using only that latest
  round's loss; lr = 3e-5, full context, stale cache (the TTS reference setting). (b) **S2 + TTS stride 10, pooled**:
  drafter context = S2; one Adam step every 10 rounds on the summed loss of all 10 rounds' valid pairs (i ≤ a+1), each
  round's term back-propagated under that round's S2 visible keys only (gradients accumulated in fp32 on the masters;
  identical to back-propagating the pooled loss, since parameters are fixed within the 10 rounds); lr = 3e-5.
  Implementation: rlsdk/adapters.py TTS(stride, pool, ctx), rlsdk/window.py grad_view; 10 CPU tests
  (tests/test_tts_ctx.py), 102/102 tests pass.
- Logged per round for every TTS variant: update_ms, grad (backward this round), step (Adam step this round);
  reported separately for full-context TTS and S2-context TTS.
- **Dominance claim (all 50 test traces):** Δ = τ_S2 − τ_TTS(stride 1), paired bootstrap 95% CI
  (scripts/paired_bootstrap.py: traces resampled with replacement within dataset, same draw for both arms,
  τ = mean of AIME and LCB pooled τ, decision variant, B = 10,000, seed 0, percentile interval).
  **"S2 dominates TTS" if the CI lower bound is ≥ 0.**
- **Hybrid claim (16 traces):** S2 + TTS(stride 10, pooled) beats TTS(stride 1) on τ with the paired-bootstrap 95% CI
  excluding 0 (lower bound > 0), **and** its mean update ms per round is ≤ 10% of TTS(stride 1)'s (both measured in
  the same run setup, mean over all rounds of the 16 traces).
- Noted before running (A30 probe, aime24_II_14, first 1,200 rounds, scratch output): update ms/round — TTS stride 1
  full context 58.0; TTS stride 10 full context 6.8 (58.1 on gradient rounds); S2 + TTS stride 1 55.7; S2 + TTS
  stride 10 pooled 42.0. The per-round backward cost is nearly independent of context length, so the pooled
  hybrid (a backward every round) is expected to fail the ≤ 10% cost criterion (42.0 / 58.0 = 72%) as implemented.
  τ seen in the probe (one trace, 1,200 rounds): 4.121 / 4.023 / 4.124 / 4.207 in the same order.

| # | start (UTC) | end (UTC) | command | A30-h | cum A30-h | note |
|---|---|---|---|---|---|---|
| 29 | 2026-10-10 10:38:09 | 11:00:17 | instance idle (awaiting instructions) | 0.369 | 2.313 | counted: instance was running |
| 30 | 2026-10-10 11:00:17 | 11:12:04 | coding on the instance: window/S2-aware TTS gradient forward, TTS stride/pool, 10 CPU tests (102/102 pass), paired bootstrap script, A30 cost probe of 4 TTS variants (aime24_II_14, 1,200 rounds each), P3 pre-registration | 0.196 | 2.509 | probe: pooled S2 + TTS stride 10 = 42 ms/round update vs 58 for TTS stride 1 (72%) → ≤ 10% cost criterion expected to fail as implemented |

**A30 session total (instance time incl. coding and idle, 08:41:09–11:12:04 UTC): 2.515 A30-h.**

**Status:** P3 arms coded and pre-registered; no held-out run launched. Open: tier plan (P1–P4, C2 seed 1, 8 A30-h cap) not in the repo; hybrid cost criterion vs probe.

## Held-out plan — PRE-REGISTERED (2026-10-10 11:26 UTC, before launch; supersedes the earlier T1/T2/P3 drafts)

**T0 (TTS setting).** AIME dev grid, 5 traces (aime24_II_02, aime24_II_09, aime24_II_13, aime24_II_14, aime24_I_10),
pooled τ, truncated: baseline 3.258; TTS lr 1e-5 **4.487**, 3e-5 4.425, 1e-4 3.596. Chosen lr = **1e-5** (best stale lr,
brief rule). Stale vs refresh: **refresh not run** (cancelled at the earlier pause) → **stale cache** (brief default).
TTS setting = lr 1e-5, stale, full context.

**T1 — all 50 test traces, in this order** (scripts/run_heldout.sh; replay, A30, 2 light workers then 1 TTS worker):
1. baseline; window (W 1024 + prompt); S2 (W 1024 + prompt + 16 far 32-token chunks, target layers 17/25/33 at t'−1);
   C1 (plain window W 1536 + prompt); C2 (W 1024 + prompt + 16 random far chunks, seed 0).
2. Gate: auto-continue only if τ_window − τ_base ≥ 0.4 on test (decision variant). Otherwise stop, report, pause.
3. TTS stride 1 (TTS setting).

**T2 — 16 test traces** (8 AIME + 8 LCB, drawn with random.Random(16) from the non-dropped test traces; file
configs/t2_traces.txt): aime24_II_01 aime24_II_10 aime24_II_12 aime24_I_01 aime24_I_03 aime24_I_04 aime24_I_05
aime24_I_14 lcb_3697 lcb_abc391_e lcb_abc391_g lcb_abc396_e lcb_arc190_c lcb_arc192_b lcb_arc192_e lcb_arc194_d.
- TTS stride 10 (TTS paper protocol: one step every 10 rounds, latest round's loss only; lr 1e-5, stale, full context);
- S2 + TTS stride 10, pooled (S2 context; one step every 10 rounds on the summed loss of all 10 rounds' valid pairs
  i ≤ a+1, backward attending only to S2's kept keys; lr 1e-5);
- S2 + TTS stride 1 (S2 context; lr 1e-5).
Dropped: window + TTS, C2 seed 1.

**Budget:** this run capped at 9 A30-h from launch. Projection: T1 light ≈ 1.4 h, gate, T1 TTS ≈ 3.2–4.0 h,
T2 ≈ 2.0 h (TTS stride 10 ≈ 0.2, hybrid ≈ 0.65, S2 + TTS stride 1 ≈ 0.85, teacher-forced passes ≈ 0.3), isolated
timing + report + 10-min grace ≈ 0.35 h → **≈ 7.3 A30-h (range 6.5–8.3)**. Cut order if the projection goes over:
S2 + TTS stride 1, then C2, then C1 (the driver re-projects before S2 + TTS stride 1 from the measured hybrid cost and
skips it if elapsed + estimate + 0.35 h > 9 h). Never cut: baseline, window, S2, TTS stride 1, TTS stride 10, hybrid.

**Statistics** (scripts/heldout_report.py): paired bootstrap over traces, stratified by dataset, same draw for all
arms, 10,000 resamples, seed 0, 95% percentile CIs; τ = mean over AIME and LCB of pooled τ, decision variant (flagged
traces truncated at loop onset; dropped traces excluded). On the 50 traces: window − base, S2 − base, TTS − base;
R_window and R_S2 = ratio of mean gains (τ_arm − τ_base)/(τ_TTS − τ_base); S2 − window, S2 − C1, S2 − C2;
S2 − TTS(stride 1). On the 16: hybrid − TTS(stride 1); (S2 + TTS stride 1) − S2 (also TTS stride 10 − stride 1,
descriptive).

**Claims (fixed now):**
- Headline: **window R ≥ 0.5** (point estimate; CI reported).
- **S2 adds value:** S2 − window ≥ 0.05 τ with CI excluding 0 (lower bound > 0), AND the CIs of S2 − C1 and S2 − C2
  both exclude 0 (lower bounds > 0). If this fails, summary.md leads with the window.
- **S2 dominates TTS:** lower CI bound of S2 − TTS(stride 1) ≥ 0.
- **Hybrid:** hybrid − TTS(stride 1) has a CI above 0, at ≤ 10% of TTS(stride 1)'s mean update ms per round (both on
  the 16 traces, each from a TTS process alone on the GPU). Note (from the probe above): expected to fail the cost
  part as implemented (≈ 72%).
- **Residual after S2:** (S2 + TTS stride 1) − S2, report only, no threshold.

**Logging:** per-round update ms + grad/step flags for every TTS variant (full context vs S2 context); S2's extra
target-side cost from bytes read (one query row vs the K cache of layers 17/25/33: 3 × 8 × 128 × 2 B × (t'−1) per
round, at 933 GB/s), plus the microbenchmark; drafting ms from isolated runs (baseline, window, S2; one process alone;
dev traces aime24_II_14 and lcb_abc393_d). τ, per-dataset τ, τ < 2K and τ ≥ 8K for every arm (also exclusion and
loops-included variants).

**When done:** report (results/heldout.md), held-out section at the top of summary.md, loop-check proposal
(results/loop_check_proposal.md, not run), commit, push, 10-min grace (logs/heldout/HOLD extends it), pause via the
JarvisLabs API (scripts/jl_pause.py; token read from /home/.jarvislabs_token, never printed/logged/committed;
read-only API check passed for instance 534259). The driver retries a failed phase once with one worker; a hard stop
or second failure → report what exists, commit, push, pause.

| # | start (UTC) | end (UTC) | command | A30-h | cum A30-h | note |
|---|---|---|---|---|---|---|
| 30b | 2026-10-10 11:12:04 | 11:19:34 | instance idle (awaiting instructions) | 0.125 | 2.634 |  |
| 31b | 2026-10-10 11:19:34 | 11:26:46 | T0 report, JarvisLabs SDK (env-jl) + read-only API check, held-out driver + report + bootstrap, tiny dry-runs ×2 (gate-stop and full path), T2 trace draw, pre-registration | 0.120 | 2.754 |  |

**Before held-out launch: session total 2.760 A30-h. The held-out run (row 32, written by the driver) is capped at 9 A30-h from its own start.**

## Addendum to the held-out plan — PRE-REGISTERED (2026-10-10 11:34 UTC; held-out run already in T1, unaffected)

**A. Hybrid cost claim — amortized.** The cost comparison uses amortized update ms per round = total update time ÷
total rounds (a stride-10 run is charged for its one update spread over 10 rounds). This is the "update ms/round"
column of scripts/heldout_report.py (mean of per-round update_ms over all rounds, including rounds without a
backward); the pooled hybrid does a backward every round, so its amortized cost includes all of them.

**B. Verifier-Prior Attention (VPA)** — runs after T2; tuned on dev only.
- Kept context: prompt + last W0 = 1024 positions + top M far 32-token chunks, chosen by the same signal as S2 (target
  layers 17/25/33, last committed query t'−1, summed over heads; chunk ranking = S2's chunk scores). Kept keys
  renumbered contiguously in original order (as S2).
- Bias on the drafter's attention logits for kept CONTEXT keys only (all drafter layers and heads; block keys 0):
  b_j = β·log(m_j + ε) − λ·log(1 + max(0, d_j − W0)/W0), m_j = target attention mass on context position j (row t'−1,
  summed over the 32 heads, mean over layers 17/25/33, normalized to sum 1 over all context positions 0..t'−1; computed
  each round from the target's q row and K cache, read through the leakage guard), ε = 1/N_ctx (N_ctx = t'), d_j = t' − p_j
  (original distance from the anchor t' to key j); prompt and recent-window keys get no distance term. b is centered
  over the kept context keys (mean subtracted).
- Unit tests: β = λ = 0, M = 16 → S2 bit-exact; β = λ = 0, M = 0 → window bit-exact; adding a constant to b changes
  nothing; block keys never biased (plus: m_j = eager attention mass; biased forward = fresh forward with the same
  additive mask).
- Step 1 (screen, 4 reference traces): M = 64 × (β, λ) ∈ {(0,0), (0.5,0), (0.5,0.5), (0.5,1), (1,0), (1,0.5), (1,1)};
  pick the best by mean τ (decision variant, mean of AIME and LCB pooled τ).
- Step 2 (confirm, the 7 remaining non-dropped dev traces: aime24_II_02, aime24_II_09, aime24_II_13, aime24_I_13,
  lcb_3701, lcb_arc193_d, lcb_arc195_c): window, S2 and the chosen VPA config. **Promote VPA to test only if**
  VPA − S2 ≥ +0.05 τ (mean) AND VPA ≥ S2 (per-trace τ, decision variant) on ≥ 75% of the 7 traces (≥ 6 of 7).
  If promoted: freeze the config in configs/frozen.yaml and commit it before any test run.
- **T3 (only if promoted):** VPA on all 50 test traces; paired bootstrap 95% CIs (same procedure as the held-out
  report) for VPA − S2, VPA − window, VPA − TTS(stride 1). Claims: "VPA improves on S2" if the CI for VPA − S2
  excludes 0; "VPA dominates TTS" if the lower CI bound of VPA − TTS(stride 1) ≥ 0.
- If not promoted: report the dev results as is (window and S2 are limits of the VPA family).
- Budget: ≈ 1 A30-h for steps 1–2, ≈ 0.5 for T3, separate from the 9 A30-h held-out cap. Unattended rule: if the
  projection exceeds 2 A30-h, stop before the next step, report, commit, push and pause (instead of asking).
- Reported per config: τ, τ AIME, τ LCB, τ ≥ 8K, drafting ms/round. VPA results go into summary.md after the
  held-out section.
- Execution: VPA code is developed in a separate worktree (branch vpa) so the running held-out test imports
  unchanged code; scripts/run_vpa.sh waits for the held-out driver to commit and push (GRACE), takes over the pause,
  merges branch vpa, runs steps 1–2 (+ T3 if promoted), reports, commits, pushes and pauses via the API. It runs only
  if the held-out run completed (not after a gate stop or failure).

| # | start (UTC) | end (UTC) | command | A30-h | note |
|---|---|---|---|---|---|
| 32 | 2026-10-10 11:26:54 | 17:04:28 (before grace + pause) | `bash scripts/run_heldout.sh` | 5.626 | complete; phases: t1_light rc=0 elapsed_h=1.059;t1_tts rc=0 elapsed_h=4.051;t2a rc=0 elapsed_h=4.856;t2b rc=0 elapsed_h=5.574;iso rc=0 elapsed_h=5.622; |

| # | start (UTC) | end (UTC) | command | A30-h | note |
|---|---|---|---|---|---|
| 33 | 2026-10-10 17:05:15 | 17:29:05 (before grace + pause) | `bash scripts/run_vpa.sh` | 0.397 | not promoted |

Row 33 note: the driver's own commit failed (a git-add pathspec for absent T3 files matched nothing); the VPA files were
committed by hand at 17:29:38 UTC (fe5ca9f) during the grace window.

**A30 session total: 08:41:09 → 17:29:38 UTC = 8.808 A30-h, plus the ≈ 10-min VPA grace before the API pause
(≈ 8.98 A30-h at pause).** Held-out run 5.626 A30-h (cap 9); VPA 0.397 A30-h (budget 2). Old-GPU total stays 2.991 GPU-h.

**Final status:** held-out claims — headline window R = 0.718 ≥ 0.5 YES; S2 adds value YES; S2 dominates TTS NO
(S2 − TTS −0.139 [−0.173, −0.107]); hybrid NO (τ +0.130 [+0.063, +0.176] but update cost 61.8% > 10%). VPA not
promoted (bias hurt monotonically; VPA − S2 = −0.049 on dev, 1/7 traces). Loop-check proposal: results/loop_check_proposal.md (not run).

## Session 3 (A30, resumed 2026-10-11 03:39:37 UTC) — thinking-mode calibration + batched pooled TTS

Budget: **5 A30-h for this session, from the resume at 03:39:37 UTC, idle time included.** Stop after any step if the
projection exceeds it. Old totals unchanged (2.991 old-GPU h; 8.98 A30-h last session).

**PRE-REGISTERED (2026-10-11 04:05 UTC, before any GPU run this session).**

Motivation (given): on the 49 test traces, 76% of the extra rounds S2 needs vs TTS fall in the first 4K output tokens
(0–2K 31%, 2–4K 45%), inside the drafter's training length. Hypothesis: most of it is a constant thinking-mode
mismatch (drafter trained with thinking off), fixable once offline.

**Step 1 — exact batched pooled update** (`rlsdk/batched.py`, TTS config `batched: true`). Each round stores its new
context rows (target features), block embeddings + query positions, indices of its older visible context rows in the
append-only WindowCache buffers (stale K/V exactly as seen, never rebuilt), its RoPE cos/sin, and the valid-slot
target logits in bf16; at the Adam step, one padded + masked batched forward/backward. CPU tests
(`tests/test_batched.py`, tiny model, fp32): gradient = per-round accumulated gradient at every step, max rel. error
9.4e-7 (stride 4, 46 steps), 6.0e-7 (stride 10), 4.0e-7 (stride 20) < 1e-3; buffers append-only; negative control
(older K/V rebuilt with current weights) breaks equality after the first step (> 1e-2); same trajectory and total
weight change as the existing pooled path. 118/118 tests pass.
- Run: S2 + TTS lr 1e-5, stride 10, pooled, batched on the 16 T2 traces (configs/t2_traces.txt), one TTS process alone
  on the GPU. **Check: τ within 0.02 of 4.429** (existing per-round pooled hybrid; decision variant). A miss = an
  implementation problem → stop and investigate before Step 3. Report amortized update ms/round (total update time ÷
  rounds) and its ratio to TTS stride 1 (69.60 ms on these 16 traces); the ≤ 10% cost criterion is NOT loosened.
- Exploratory (cost vs τ curve only, no decision): batched stride 20, same 16 traces, same process. "full" stays at
  stride 10 (fixed now; a stride chosen on test traces would bias the test).

**Step 2 — calibration traces**, disjoint from all 62 dev/test problems: AIME 2025 (`MathArena/aime_2025`, 30) + 30 LCB
v6 hard drawn from the 48 unused hard problems of test6.jsonl (sorted by question_id, minus the 32 used;
`random.Random(0).sample(unused, 30)`). Same templates (DFlash bb48ef9), Qwen3 chat template, thinking on, greedy,
max 8,192 new tokens, vLLM 0.31. Loop flag + cut rule as Amendment 2 / §12.4 (dropped if < 1,024 output tokens
remain). Check problem texts are disjoint from the 62.

**Step 3 — one-time calibration** (offline, from the released DFlash weights). For each calibration trace (truncated
at the loop cut): one teacher-forced target pass (context features of layers 1/9/17/25/33, h*, and the S2 attention
rows of layers 17/25/33); features held in CPU RAM for the run, never written to disk. Training sample = an anchor t
drawn uniformly from output positions with t + 15 ≤ last position (seed 0); drafter input in S2 layout at frontier t:
prompt + last 1024 context positions + 16 far 32-token chunks scored by row t−1 (target layers 17/25/33), contiguous
positions, block [s_t, MASK×15]; context K/V computed in-graph with the current weights (teacher-forced). Loss =
Σ_{i=1..15} w_i KL(softmax(lm_head h*_{t+i−1}) ‖ softmax(lm_head z_i)), w_i = exp(−(i−1)/7). Adam (0.9, 0.999,
ε 1e-8, no weight decay, constant lr) on fp32 masters, bf16 drafting copy. lr ∈ {3e-6, 1e-5}, same anchors and order
for both. Number of steps fixed by a timing-only probe (no evaluation) so both lrs + teacher-forced passes fit in
≤ 0.75 A30-h; logged before training. **lr chosen by S2+cal τ on the 4 reference traces** (static S2 replay with the
calibrated weights); checkpoint frozen, its sha256 committed (weights under ckpt/, not in git).

**Step 4 — screen, 4 reference traces** (aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d), all arms on the A30:
S2 (existing A30 run); TTS stride 1 lr 1e-5 full context stale (new, all 4 traces); S2+cal (static); full = S2+cal +
batched pooled TTS stride 10 lr 1e-5 (TTS state starts from the calibrated weights each trace). τ (decision variant)
overall and per output-position bin 0–2K, 2–4K, 4–8K, ≥ 8K. Readings (fixed):
- Mode mismatch supported if S2+cal closes ≥ 50% of S2's 0–4K τ deficit vs TTS:
  (τ_S2cal − τ_S2)/(τ_TTS − τ_S2) on rounds starting at output position < 4096 ≥ 0.5 (deficit ≤ 0 → not supported).
- Promote "full" to test if τ_full ≥ τ_TTS + 0.10 and full beats TTS on ≥ 3 of 4 traces. Promote S2+cal if τ_S2cal ≥ τ_TTS.
- If mode mismatch is NOT supported, add one arm before deciding: S2 + TTS lr 1e-5 (released weights) with stride 1
  for the first 2K output tokens, then batched pooled stride 10; it is promoted under the same rule as "full".
- After the screen: post the table here. If any promotion rule triggers → Step 5 automatically; else stop, report,
  pause.
- Selection optimism: the lr for S2+cal is chosen on these same 4 traces, so S2+cal and full are optimistic here;
  TTS's lr (1e-5) was also chosen on dev, so both sides are tuned the same way. Step 5 is the unbiased test.

**Step 5 — held-out, promoted arms only, 49 test traces** (decision variant). Paired bootstrap 95% CIs (stratified by
dataset, B = 10,000, seed 0, percentile; scripts/heldout_report.py procedure) vs TTS stride 1 (existing runs) and vs
S2 (existing). "Beats TTS": lower CI bound of (arm − TTS) > 0. "Cheaper": amortized update ms/round ≤ 10% of TTS
stride 1's on the 49 traces (S2+cal: 0 by construction); arm timed in a process alone on the GPU. τ per bin for every
arm.

**Step 6 — fairness, if budget allows:** TTS stride 1 lr 1e-5 with full context-cache refresh after every step vs
stale, 4 reference traces. If refresh beats stale by > 2% τ: stop and report before any "beats TTS" claim. If Step 6
is not run, summary.md states that the TTS comparison uses the stale cache and the refresh check was not run, and
every "beats TTS" claim carries that caveat.

| # | start (UTC) | end (UTC) | command | A30-h | cum A30-h | note |
|---|---|---|---|---|---|---|
| 34 | 2026-10-11 03:39:37 | 04:05 | resume, orientation, env checks (112/112 tests), awaiting go-ahead, Step 1 code + 6 CPU tests, resume.sh dflash check (GPU path), pre-registration | 0.42 | 0.42 | idle + coding counted |

Notes added before any Step 1 result was read (2026-10-11 04:33 UTC):
- Real-model exactness check of the batched update (`scripts/batched_check_fp32.py`, aime24_II_01, 450 rounds): with
  the drafter in fp32, batched vs per-round accumulated gradient max rel. error 1.6e-4 (median 1.9e-5; 15 steps with
  far chunks active, max 9.9e-5). In bf16 (the run setting) the same comparison gives median 2.8%, max 5.6%: bf16
  rounding of one 10-round gradient vs ten separately rounded gradients, not an implementation error
  (`results/batched_check_fp32.json`). The τ check above remains the run-level criterion.
- Step 2 data hygiene (before generation): `lcb_abc388_g` (drawn) shares its statement with test problem
  `lcb_abc388_e` (harder variant of the same task) → replaced by the next draw of the same seeded sample
  (`lcb_abc396_f`); `rlsdk/data.py::CAL_EXCLUDE`. Max 8-gram Jaccard of any calibration prompt vs the 62 = 0.16 (template
  phrase). List: configs/cal_traces.txt.
- Step 3 step count: fixed automatically by `rlsdk.calibrate --budget-h 0.75` (20-step timing probe, discarded) and
  logged as "STEPS FIXED" in ckpt/cal/train_log.txt before training; batch 16 anchors/step.
- Step 4 contingency arm implemented in advance (TTS config `warm: 2048`; tests/test_batched.py), used only if the
  mode-mismatch reading is "not supported". 124/124 CPU tests pass.

**Step 1 OOM and fix (2026-10-11 04:45 UTC; no Step 1 τ has been read or reported yet).** The first Step 1 run
(04:21:11–04:37:41) hit CUDA OOM in the batched backward of the stride-20 job on lcb_abc391_e (19 of 32 jobs done;
peak reserved had grown to 18–20 GB at stride 10 and 22.7 GB at stride 20 on LCB traces). Cause: the batched forward
holds the GQA-expanded K/V (and math-SDPA fp32 copies) of all pooled rounds at once (≈ 0.4–0.5 GB per round). Fix
(same maths): the step's rounds are run as micro-batches of 5 rounds (forward + backward per micro-batch, gradients
summed in fp32 on the masters; TTS config `micro`, default 5); tests/test_batched.py checks micro ∈ {3, 5, 10}
(all < 1e-3). For a clean cost measurement, all 16 stride-10 jobs are rerun with micro-batching (the 10 finished
full-batch stride-10 files are kept in results/raw_test_s3_nomicro/ for the record, not used). Stride 20
(exploratory): the 9 finished full-batch files are kept and the remaining 7 traces run micro-batched (disclosed in
the report). The Step 1 rerun runs after the calibration evaluations (scripts/run_s3_after_cal.sh), which stops
before Step 4 if the τ check fails.

**Eval OOM (05:33–05:40 UTC):** the two S2+cal evaluation workers (lr pick) ran next to the TTS stride 1 reference
worker and OOMed in their teacher-forced passes (TTS worker ≈ 13 GB steady + a light worker's TF pass > 24 GB); one
S2+cal file (lr 3e-6, aime24_II_14) had finished and is kept (deterministic replay). The evals are rerun after the TTS
reference worker ends, as 2 light workers with no TTS worker alongside (scripts/run_s3_after_cal.sh), then Step 1
rerun, τ guard, lr pick, Step 4. Step 3 itself: 04:54:08–05:32:52 = 0.646 A30-h (cap 0.75); 59 traces (1 dropped by the
loop rule), 1,415 steps × 16 anchors per lr; ckpt sha256: lr 3e-6 7a0e0fb8d7002feb6f0f51f7c0465733f67157c628bffae5678bd3461021ea0d,
lr 1e-5 cc25cc15a3782b56e2a5be9d1a17bbaf74718be966413447b76a5b728e8ca5d7.

**Second Step 1 OOM and fix (06:20 UTC; Step 1 τ still not read).** The micro-batched rerun (05:52:28–06:20) also hit
OOM in the batched forward, on lcb_arc190_c (prompt 1,689 tokens → ≈ 3,240 visible keys per round). Root cause: the
batched forward went through transformers' SDPA path, which expands K/V to all 32 query heads (repeat_kv) and, in the
math kernel, keeps fp32 copies for the backward. Fix (same maths): the batched forward uses its own grouped-query
attention (`rlsdk/batched.py::gqa_attention`: per KV head its 4 query heads, fp32 scores/softmax, masked keys −inf;
no K/V expansion); tests: equals transformers' SDPA output, and the batched gradient still equals the per-round
(SDPA) accumulated gradient (< 1e-3, micro ∈ {3, 5, 10}); 127/127 CPU tests pass. Micro-batch default → 10 rounds (one batched
pass per stride-10 step). All 16 stride-10 jobs rerun with this implementation (the micro-5 files are kept in
results/raw_test_s3_micro5/, not used). Budget trim: stride 20 (exploratory) is not rerun; 15/16 traces exist
(9 full-batch, 6 micro-5, all same maths; lcb_arc190_c missing) and are reported as a partial, exploratory table.
Step 6 (TTS stride 1 refresh, 4 reference traces) runs now (06:21) on the otherwise idle GPU; chain driver
(scripts/run_s3_chain.sh): Step 6 → Step 1 rerun → τ guard → lr pick → Step 4.

**Step 6 result (06:21–06:40:53 UTC, run early on the idle GPU; pre-registered rule):** TTS stride 1 lr 1e-5, 4
reference traces, A30: stale τ 4.319, refresh (full context-cache rebuild after every step) τ 4.316 → refresh − stale
= −0.07% (stop threshold +2%) → **no stop; stale TTS is a fair baseline.** Per trace (stale / refresh): aime24_II_14
5.024 / 4.989, aime24_I_10 4.200 / 4.200, lcb_abc393_d 3.816 / 3.836, lcb_arc191_d 4.236 / 4.236 (refresh is active:
trajectories diverge from round 45–121 on; equal τ on two traces with equal round counts is coincidence). Refresh
costs +12–14 ms/round of update.

**Step 4 screen — posted before continuing (screen 06:56:12–07:00:34 UTC; lr 1e-5 picked: S2+cal τ 4.540 vs 4.468 at 3e-6).**

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

Step 1 result (06:40:59–06:56:07): batched stride 10 τ 4.426 vs per-round pooled 4.429 → τ check PASS; amortized update 11.49 ms/round = 16.5% of TTS stride 1 (69.60) → cost criterion NOT met; batched s10 − TTS s1 +0.127 [+0.060, +0.173] (16 traces). results/batched.md.

Budget at 07:15 UTC: 3.59 A30-h used (incl. 07:00–07:15 idle: monitor had expired). Step 5 both arms ≈ 1.27 A30-h → projection ≈ 5.0–5.1 with wrap-up > 5 cap. Per the budget rule: Step 5 runs S2+cal first (≈ 0.5 A30-h, fits); "full" on test only if the budget is extended.

**Budget extended by the user (07:20 UTC): ≈ 6 A30-h for this session** — "full" runs on all test traces after S2+cal (same calibrated drafter, one process alone on the GPU).

**Step 5 (held-out, 49 test traces):** S2+cal 07:15:33–07:39:07 (0.393 A30-h); full 07:39:07–08:03 (OOM in a
teacher-forced pass after 28 traces: previous jobs' memory not released; fix: `gc.collect()` before each trace's TF
pass in rlsdk/pool.py) + 08:04:21–08:27:37 (remaining 22; each process alone on the GPU). Results
(results/s3_heldout.md): **S2+cal − TTS stride 1 +0.200 [+0.179, +0.222] → beats TTS; cheaper (0 ms) → YES.**
**full − TTS +0.340 [+0.319, +0.360] → beats TTS; update 11.36 ms/round = 16.3% → not cheaper.** S2+cal − S2 +0.339
[+0.308, +0.371]; full − S2 +0.479 [+0.440, +0.520]; full − S2+cal +0.140 [+0.122, +0.157]. Step 6 (refresh − stale
−0.07%) was run, so no stale-cache caveat applies. Reference traces in the test set: none (all dev; asserted).

| # | start (UTC) | end (UTC) | what | A30-h | cum A30-h |
|---|---|---|---|---|---|
| 34 | 03:39:37 | 04:21:11 | resume, orientation, env checks, awaiting go-ahead, Step 1 code + tests, bf16/fp32 GPU probes, pre-registration | 0.693 | 0.693 |
| 35 | 04:21:11 | 04:37:41 | Step 1 run #1 (OOM at stride 20; superseded) | 0.275 | 0.968 |
| 36 | 04:37:50 | 04:54:08 | Step 2: 60 calibration traces (vLLM) | 0.272 | 1.242 |
| 37 | 04:54:08 | 05:32:52 | Step 3: calibration, both lrs | 0.646 | 1.887 |
| 38 | 05:32:52 | 05:52:28 | TTS stride 1 lr 1e-5 on the 4 reference traces; S2+cal evals (OOM, rerun 05:49:48–05:52:28) | 0.327 | 2.215 |
| 39 | 05:52:28 | 06:21:31 | Step 1 rerun #2 (micro 5; OOM; superseded) + fix | 0.484 | 2.699 |
| 40 | 06:21:31 | 06:40:53 | Step 6: TTS refresh, 4 reference traces | 0.323 | 3.021 |
| 41 | 06:40:59 | 06:56:07 | Step 1 final (stride 10, 16 traces) | 0.252 | 3.276 |
| 42 | 06:56:12 | 07:00:34 | Step 4: full on the 4 reference traces + screen report | 0.074 | 3.350 |
| 43 | 07:00:34 | 07:15:33 | idle (expired monitor) + RUNLOG table, budget check | 0.250 | 3.600 |
| 44 | 07:15:33 | 08:27:37 | Step 5: S2+cal and full on the test traces (incl. the OOM restart) | 1.201 | 4.800 |
| 45 | 08:27:37 | ≈ 08:33 | reports, summary.md, HANDOFF.md, NOTES (SharpDraft paper read, related work), commit, push, API pause | 0.09 | ≈ 4.89 |

Cumulative = wall-clock instance time since the resume (short gaps between rows included).

**Session 3 total: 03:39:37 → API pause ≈ 08:33 UTC = ≈ 4.89 A30-h** (cap 5, extended to ≈ 6 by the user). Old totals: 2.991 old-GPU h; session 2 ≈ 8.98 A30-h.

## Session 4 (A30, instance up 2026-10-11 08:36:37 UTC, MACHINE_ID 534736) — SharpDraft, thinking-mode drafter, cal-full

Budget: **6 A30-h from 08:36:37 UTC, idle included (cap 14:36:37 UTC).** Order: item 1 (SharpDraft), item 2
(thinking-mode drafter), item 3 (cal-full); items 4–6 of the user's list are out of this phase. Each item's table is
posted here, committed and pushed before the next item starts. Old totals unchanged (2.991 old-GPU h; session 2
≈ 8.98 A30-h; session 3 ≈ 4.89 A30-h).

Step 0 (08:36:37–08:45): orientation; main clean = origin/main (1524300); RESUME OK; 127/127 CPU tests; frozen S2+cal
checkpoint sha256 cc25cc15…d5ca7 matches. "full" per-bin CIs vs S2+cal computed from the saved replays (overall
reproduces s3_heldout.md exactly): full − S2+cal +0.140 [+0.122, +0.157]; 0–2K +0.023 [+0.015, +0.031], 2–4K +0.114
[+0.090, +0.140], 4–8K +0.164 [+0.144, +0.185], ≥ 8K +0.251 [+0.216, +0.284]. **"full" is reported as is** (user
decision 2026-10-11): beats TTS +0.340 but 11.36 ms/round = 16.3% of TTS's update cost → fails the pre-registered
≤ 10% rule; the 10% line is not relaxed. S2+cal stays the headline.

**PRE-REGISTERED (2026-10-11 08:55 UTC, commit 0c0ebde, before any GPU run this session).**

Fixed for every arm: the 49-trace test set, the loop-cut rule (decision variant), the bins 0–2K / 2–4K / 4–8K / ≥ 8K
(round start output position), τ = mean of AIME and LCB pooled τ, paired bootstrap 95% CIs stratified by dataset,
B = 10,000, seed 0 (scripts/heldout_report.Boot; per-dataset CIs bootstrap within the dataset). Reused, not rerun:
S2, TTS stride 1 (results/raw_test), S2+cal, full (results/raw_test_cal), baseline (results/raw_test,
results/raw_a30). Every new arm reports τ overall, per bin, per dataset, paired CIs vs TTS stride 1 and vs S2+cal,
and update ms/round (amortized; 0 for every arm below: no updates). **"Within the CI of X"** = the paired 95% CI of
(arm − X) contains 0; "beats X" = lower bound > 0; "below X" = upper bound < 0. No hardware anchor (same GPU type,
A30; user decision).

**Item 1 — SharpDraft** (arXiv 2610.05106; NOTES.md §14 re-checked against the PDF this session). Eq. (5):
r = max((N − k)/(N0 − k), 1), β̂ = min(e^0.5, 1 + log r / s), k = 32, s = 5, N0 = 2048 (the paper's fixed value for
Qwen, used as is, not tuned by us); applied after q_norm in all 5 drafter layers with the BF16 rule Eq. (18);
weights and context K/V unchanged. N = keys visible to the query = visible context rows + verify_size block slots
(16 except at the very end), counted the same way in every layout (full context: N = start + 16; S2: N = L + 16).
Code: rlsdk/adapters.py::SharpDraft, rlsdk/replay.py::scaled_queries; tests/test_sharpdraft.py (14 CPU tests: g = 1
at/below N0, formula, cap reached and held, scale 1 = baseline bit-exact in full and S2 layouts (fp32, bf16), query
scaling = logit scaling, Eq. 18 arithmetic, N counting and the applied scale per round, S2 layout AIME scale = 1).
- 1d **sanity anchor**, 4 reference traces (aime24_II_14, aime24_I_10, lcb_abc393_d, lcb_arc191_d), released
  drafter, full context, vs the existing A30 baseline (results/raw_a30, τ 3.016). **Pass** iff τ_SharpDraft >
  τ_baseline AND the gain grows with output position: Δ(4–8K) > Δ(0–2K) and Δ(≥ 8K) > Δ(0–2K). **Fail → stop item 1,
  re-read the paper, report; no test-set SharpDraft runs.**
- 1e **arm SD** (the arm that matters): SharpDraft, released drafter, full context, 49 test traces. Reading (fixed
  by the user): SD within the CI of S2+cal → our contribution is calibration + target-guided selection, not the
  window. Also reported: SD − baseline, SD − TTS stride 1, SD − S2.
- 1e **arm SD+S2+cal**: SharpDraft in the S2 layout with the frozen S2+cal checkpoint, **26 LCB test traces only**.
  On AIME β̂ = 1 on every round by construction (prompt ≤ 413 → N ≤ 413 + 1024 + 512 + 16 = 1,965 ≤ N0; asserted on
  every AIME trace and every logged AIME round of the S2+cal replays, tests/test_sharpdraft.py), so its AIME replay
  is the S2+cal replay bit for bit: the AIME rounds are taken from S2+cal and the report says so. Reading:
  complementary iff lower CI bound > 0 vs both S2+cal and SD. "SharpDraft + S2" (released weights) is dropped.
- Scale diagnostics (paper paragraph): per-round β̂ and N for SD (full context) and SD+S2+cal: mean, max, fraction of
  rounds at β̂ = 1, fraction at the cap e^0.5, and the visible-key count distribution, per dataset.
- Cost: ≈ 0.15 (anchor) + 0.7 (SD) + 0.25 (SD+S2+cal, LCB) A30-h. SD and SD+S2+cal run as two light workers side by
  side (draft ms indicative only).

**Item 2 — thinking-mode drafter** `jgeuter/qwen3-4b-dpard-thinking-b16`, pinned at commit
78baf299fc6dc793aab6c35ce44676a980e266d1 (HF, 2026-09-21; D-PARD objective, ~101K thinking-mode samples, max len
8192). Same DFlashDraftModel class, block 16, mask id 151669, hidden 2560, **3 layers, target feature layers
[1, 17, 33]** (released: 5 layers, [1, 9, 17, 25, 33]) — noted, not forced: the replay uses the drafter's own
`target_layer_ids`; no lm_head / embeddings in the checkpoint (target's are used, as for the released drafter).
S2 selection signal unchanged (target layers 17/25/33 attention). Arms on the 49 test traces: **thinking baseline**
(full context) and **thinking + S2** (W 1024, keep prompt, 16 far chunks, mid3). Reading (fixed): thinking baseline
≥ S2+cal (i.e. not below: CI upper bound of (thinking baseline − S2+cal) ≥ 0) → calibration's gain is available by
retraining; lead with S2 and the hybrid. Also reported: thinking + S2 − thinking baseline (does S2 help a
thinking-trained drafter), and vs TTS stride 1. ≈ 0.05 load check + 0.5 + 0.4 A30-h.

**Item 3 — cal-full** (gate fixed now: **start only if ≥ 2.5 A30-h remain under the 6-h cap when item 2 finishes,
i.e. item 2 done by 12:06:37 UTC; else stop, report, pause**). Calibrate the released drafter exactly as S2+cal
(traces/traces_cal.jsonl, same 59 traces and loop cuts, same anchors — seed 0, 1,415 steps × 16, lr 1e-5, Adam
(0.9, 0.999, 1e-8), fp32 masters, same DFlash-weighted KL) but in the **full-context layout**: visible rows = all
positions [0, t) at their own positions, block at t..t+15 (= the stock replay's drafter input). Micro-batching of
the 16-anchor batch is allowed if memory needs it (same gradient: mean over the 16). Then replay **without a window**
(full context) on the 49 test traces; per-trace output files, so a partial run is not wasted. Reading (fixed):
cal-full within the CI of S2+cal → the S2 layout adds nothing on top of calibration; say so. Also reported vs TTS
stride 1 and vs baseline. ≈ 1.2–1.6 fine-tune + 0.7 run A30-h.

### Item 1 — SharpDraft: results (posted 09:26 UTC, before item 2 starts)

**1d anchor (08:56:05–08:58:14, 2 light workers; results/s4_anchor.md):** released drafter, full context, 4 reference
traces: baseline τ 3.016 → SharpDraft **3.711**; Δ by bin 0–2K +0.000, 2–4K +0.078, 4–8K +0.617, ≥ 8K +1.146; better
on 4/4 traces → **PASS** (gain grows with output position). β̂ mean 1.218, max 1.420, 14.5% of rounds at 1, 0% at cap.

**1e test arms (08:58:56–09:23:25; SD and SD+S2+cal as 2 light workers, then a second SD worker;
results/s4_sharpdraft.md).** τ (decision variant, 49 test traces):

| arm | τ | τ 0–2K | τ 2–4K | τ 4–8K | τ ≥8K | τ AIME (23) | τ LCB (26) | update ms/round |
|---|---|---|---|---|---|---|---|---|
| TTS stride 1 (reused) | 4.208 | 3.693 | 4.396 | 4.213 | 4.483 | 4.353 | 4.063 | 69.63 |
| S2+cal (reused) | 4.408 | 3.779 | 4.370 | 4.578 | 4.756 | 4.607 | 4.209 | 0 |
| **SharpDraft (SD, full context)** | **3.644** | 3.517 | 3.980 | 3.630 | 3.557 | 3.889 | 3.399 | 0 |
| SD+S2+cal (LCB run; AIME = S2+cal by construction) | 4.410 | 3.781 | 4.369 | 4.579 | 4.761 | 4.607 | 4.214 | 0 |

| contrast (paired 95% CI) | overall | 0–2K | 2–4K | 4–8K | ≥8K | AIME | LCB |
|---|---|---|---|---|---|---|---|
| SD − TTS stride 1 | −0.564 [−0.624, −0.502] | −0.176 | −0.416 | −0.583 | −0.925 | −0.464 [−0.537, −0.380] | −0.664 [−0.759, −0.575] |
| SD − S2+cal | **−0.764 [−0.824, −0.702]** | −0.262 | −0.390 | −0.947 | −1.198 | −0.718 [−0.790, −0.636] | −0.810 [−0.904, −0.720] |
| SD+S2+cal − TTS stride 1 | +0.202 [+0.181, +0.224] | +0.088 | −0.027 | +0.366 | +0.279 | +0.254 [+0.225, +0.288] | +0.151 [+0.120, +0.177] |
| SD+S2+cal − S2+cal | **+0.002 [+0.001, +0.004]** | +0.001 [−0.000, +0.003] | −0.001 [−0.006, +0.005] | +0.002 [−0.001, +0.005] | +0.006 [+0.002, +0.010] | +0.000 (identical) | +0.005 [+0.001, +0.008] |
| SD − baseline (reused) | +0.544 [+0.478, +0.594] | | | | | | |
| SD − S2 (reused, released weights) | −0.425 [−0.467, −0.374] | | | | | | |
| SD+S2+cal − SD | +0.766 [+0.705, +0.826] | | | | | | |

(per-bin CIs of every contrast: results/s4_sharpdraft.md.)

Scale diagnostics (decision-variant rounds): **SD, full context:** β̂ mean 1.202 (AIME 1.193, LCB 1.209), max 1.438,
16.1% of rounds at β̂ = 1, **0% at the cap** e^0.5; visible keys N median 6,103, p90 13,216, max 18,073.
**SD+S2+cal, S2 layout (LCB):** β̂ mean 1.014, max 1.093, 42.9% at 1, 0% at cap; N median 2,126, p90 2,426, max 3,241.
**AIME, S2 layout: β̂ = 1 on 100% of rounds** (N ≤ 1,965 < N0, by construction; unit-tested).

Readings (pre-registered):
- **SD vs S2+cal: below** (−0.764, CI excludes 0) → the "within the CI" reading does not apply. SharpDraft recovers
  part of the full-context decay (+0.544 over the released baseline, growing with position), but S2 alone (released
  weights) is +0.425 above it and S2+cal +0.764.
- **SD+S2+cal: the pre-registered "complementary" rule is met** (lower CI > 0 vs S2+cal and vs SD), **but the size is
  +0.002 τ (0.05%)**: zero on AIME by construction, +0.005 on LCB, where β̂ averages 1.014. Reported as complementary
  per the rule, with that magnitude stated every time; in practice S2 removes most of the dilution SharpDraft
  corrects for (N stays ≈ N0).
- "SharpDraft + S2" (released weights) dropped (user decision).
