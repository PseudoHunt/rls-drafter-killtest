#!/bin/bash
# Amendment §12.3 benchmark: alone (no MPS) vs 6 concurrent workers under MPS; 1500 rounds per job.
set -e
cd /home/rls_drafter_killtest
export HF_HOME=$PWD/hf_cache PYTHONPATH=$PWD
PY=/home/envs/env-dflash/bin/python
B=results/bench; rm -rf $B; mkdir -p $B
COMMON="--traces traces/traces.jsonl --max-rounds 1500 --tf-lock $B/.tf.lock"
echo "START $(date -u +%FT%TZ)"
t0=$(date +%s.%N)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_rls.yaml --out $B/alone --ids lcb_3701
t1=$(date +%s.%N)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_tts.yaml --out $B/alone --ids lcb_3701
t2=$(date +%s.%N)
echo "ALONE_RLS_WALL $(awk "BEGIN{print $t1-$t0}")  ALONE_TTS_WALL $(awk "BEGIN{print $t2-$t1}")"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS_STARTED"
PIDS=()
t3=$(date +%s.%N)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_rls.yaml  --out $B/mps_w1 --ids lcb_3701 > $B/w1.log 2>&1 &
PIDS+=($!)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_tts.yaml  --out $B/mps_w2 --ids lcb_3701 > $B/w2.log 2>&1 &
PIDS+=($!)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_tts.yaml  --out $B/mps_w3 --ids aime24_II_02 > $B/w3.log 2>&1 &
PIDS+=($!)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_rls.yaml  --out $B/mps_w4 --ids aime24_I_13 > $B/w4.log 2>&1 &
PIDS+=($!)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_base.yaml --out $B/mps_w5 --ids lcb_arc191_d > $B/w5.log 2>&1 &
PIDS+=($!)
$PY -m rlsdk.pool $COMMON --plan configs/plan_bench_base.yaml --out $B/mps_w6 --ids lcb_abc393_d > $B/w6.log 2>&1 &
PIDS+=($!)
( while sleep 5; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; done ) > $B/mem.log &
MON=$!
wait ${PIDS[@]} || true
kill $MON
t4=$(date +%s.%N)
echo "MPS_WALL $(awk "BEGIN{print $t4-$t3}")"
grep -h "tau=\|Traceback\|Error" $B/w*.log || true
echo "END $(date -u +%FT%TZ)"
