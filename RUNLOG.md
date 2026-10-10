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
