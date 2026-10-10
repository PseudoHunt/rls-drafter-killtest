#!/bin/bash
# TTS determinism check: alone x2 (no MPS) + 2 concurrent under MPS, 600 rounds, lcb_3701.
cd /home/rls_drafter_killtest
export HF_HOME=$PWD/hf_cache PYTHONPATH=$PWD
PY=/home/envs/env-dflash/bin/python
B=results/bench_det; rm -rf $B; mkdir -p $B
C="--traces traces/traces.jsonl --max-rounds 600 --ids lcb_3701 --plan configs/plan_bench_tts.yaml --tf-lock $B/.tf.lock"
echo "START $(date -u +%FT%TZ)"
$PY -m rlsdk.pool $C --out $B/alone1
$PY -m rlsdk.pool $C --out $B/alone2
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
$PY -m rlsdk.pool $C --out $B/mps1 & P1=$!
$PY -m rlsdk.pool $C --out $B/mps2 & P2=$!
wait $P1 $P2
echo "END $(date -u +%FT%TZ)"
