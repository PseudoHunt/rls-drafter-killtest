#!/bin/bash
# Stage 3 (dev). Phases run sequentially; MPS concurrency inside phases. Resumable: finished files are skipped.
cd /home/rls_drafter_killtest
export HF_HOME=$PWD/hf_cache PYTHONPATH=$PWD CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
PY=/home/envs/env-dflash/bin/python
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY logs/stage3
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
LOCK=results/.tf.lock
ts() { date -u +%FT%TZ; }

pool() {  # pool NAME N_WORKERS KIND PLAN OUT TRACES
  local name=$1 n=$2 kind=$3 plan=$4 out=$5 traces=$6; local pids=() rc=0
  echo "PHASE_START $name $(ts)"
  for i in $(seq 0 $((n-1))); do
    $PY -m rlsdk.pool --traces $traces --split dev --plan $plan --out $out --kind $kind \
        --tf-lock $LOCK --worker-index $i > logs/stage3/${name}_w$i.log 2>&1 &
    pids+=($!); sleep 20   # stagger model loads
  done
  for p in ${pids[@]}; do wait $p || rc=$?; done
  echo "PHASE_END $name rc=$rc $(ts)"
  grep -h "HARD STOP\|Traceback\|OutOfMemory" logs/stage3/${name}_w*.log && { echo "STAGE3_ABORT $name"; exit 1; }
  return 0
}

$PY scripts/make_plans.py baseline
pool baseline 5 light configs/plan_dev_baseline.yaml results/raw traces/traces.jsonl
pool nothink 3 light configs/plan_dev_baseline.yaml results/raw_nothink traces/traces_nothink.jsonl

S2=$($PY scripts/stage3_helpers.py sigma2); echo "SIGMA2 $S2"
$PY scripts/make_plans.py grid --sigma2 $S2
echo "RLS_CHECKS_START $(ts)"
$PY scripts/rls_checks.py --traces traces/traces.jsonl --trace-id lcb_3701 --sigma2 $S2 --n0 16 --lam-fp64 0.995 \
    --out results/rls_checks.json > logs/stage3/rls_checks.log 2>&1 || { echo "STAGE3_ABORT rls_checks"; exit 1; }
echo "RLS_CHECKS_END $(ts)"
DT=$($PY scripts/stage3_helpers.py use_fp64); echo "RLS_DTYPE $DT"
[ "$DT" = "LAM1_FAIL" ] && { echo "STAGE3_ABORT lam1 check failed"; exit 1; }

pool grid_light 5 light configs/plan_dev_grid.yaml results/raw traces/traces.jsonl
pool grid_tts 2 heavy configs/plan_dev_grid.yaml results/raw traces/traces.jsonl

LR=$($PY scripts/stage3_helpers.py best_tts_lr); echo "BEST_STALE_LR $LR"
$PY scripts/make_plans.py refresh --lr $LR
pool refresh 2 heavy configs/plan_dev_refresh.yaml results/raw traces/traces.jsonl
W=$($PY scripts/stage3_helpers.py refresh_wins $LR); echo "REFRESH_WINS $W"
if [ "$W" = "1" ]; then
  $PY scripts/make_plans.py refresh --lr 1e-05 3e-05 0.0001
  pool refresh_all 2 heavy configs/plan_dev_refresh.yaml results/raw traces/traces.jsonl
fi
echo "STAGE3_DONE $(ts)"
