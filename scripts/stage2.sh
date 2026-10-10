#!/bin/bash
# Stage 2: real-vs-replay check (3 dev prompts, 4K) + timing probe on one dev trace.
set -e
cd /home/rls_drafter_killtest
export HF_HOME=$PWD/hf_cache PYTHONPATH=$PWD
PY=/home/envs/env-dflash/bin/python
echo "START $(date -u +%FT%TZ)"
$PY scripts/real_vs_replay.py --traces traces/traces.jsonl --out results/real_vs_replay.json
echo "RVR_DONE $(date -u +%FT%TZ)"
$PY -m rlsdk.run --traces traces/traces.jsonl --plan configs/plan_timing.yaml --out results/timing_probe \
    --split dev --ids "$TIMING_ID"
echo "END $(date -u +%FT%TZ)"
