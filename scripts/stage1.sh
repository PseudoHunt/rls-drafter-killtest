#!/bin/bash
# Stage 1: main traces (thinking ON) + Amendment-4 control (thinking OFF, 6 dev AIME).
set -e
cd /home/jl_fs/rls_drafter_killtest
export HF_HOME=$PWD/hf_cache VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "START $(date -u +%FT%TZ)"
/home/envs/env-vllm/bin/python scripts/gen_traces.py --out traces/traces.jsonl
echo "MAIN_DONE $(date -u +%FT%TZ)"
/home/envs/env-vllm/bin/python scripts/gen_traces.py --out traces/traces_nothink.jsonl --no-thinking --only-split dev --only-dataset aime
echo "END $(date -u +%FT%TZ)"
