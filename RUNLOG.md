# RUNLOG

Venvs live on local disk at `/home/envs/{env-dflash,env-vllm}` (rebuild: `uv venv -p 3.12`, then
`uv pip install "dflash[local]==0.1.0" pytest pyyaml matplotlib numpy pandas accelerate` /
`uv pip install vllm datasets pyyaml`; exact pins in results/ENV.md).
`PERSIST=/home/jl_fs/rls_drafter_killtest`, `HF_HOME=$PERSIST/hf_cache`.

| # | start (UTC) | end (UTC) | command | GPU-h | cum GPU-h | note |
|---|---|---|---|---|---|---|
| 0 | 2026-10-10 03:55 | 03:58 | env setup (uv venvs) | 0 | 0 | CPU |
| 1 | 2026-10-10 04:14 | 04:16 | `pytest -q tests` (CPU, CUDA_VISIBLE_DEVICES="") | 0 | 0 | 28 passed |
| 2 | 2026-10-10 04:31:36 | 05:00:06 | `tmux: bash scripts/stage1.sh` (vLLM 0.31, 62 thinking-ON traces + 6 thinking-OFF dev-AIME control) | 0.475 | 0.475 | 62 traces 1192 s + control 196 s + 2×model load; 17/62 flagged looping → §9 stop |
