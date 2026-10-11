#!/bin/bash
# Session 4, item 1e: SD (released drafter, full context, 49 test traces) and SD+S2+cal (frozen S2+cal ckpt, S2
# layout, 26 LCB test traces). Two light workers; the second turns into an SD worker when the LCB arm is done.
source /home/rls_drafter_killtest/scripts/resume.sh >/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOCK=results/.s4_tf.lock
LCB=$(python -c "import json;print(' '.join(json.loads(l)['trace_id'] for l in open('traces/traces.jsonl') if json.loads(l)['split']=='test' and json.loads(l)['dataset']=='lcb'))")
echo "ITEM1_START $(date -u '+%F %T')"
python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_s4_sd.yaml --out results/raw_test_s4 \
  --tf-lock $LOCK --worker-index 0 > logs/s4/sd_w0.log 2>&1 &
( python -m rlsdk.pool --traces traces/traces.jsonl --ids $LCB --plan configs/plan_s4_sd_s2.yaml \
    --out results/raw_test_cal_s4 --draft ckpt/cal/lr1e-05 --tf-lock $LOCK > logs/s4/sdcal_w.log 2>&1
  echo "SDCAL_END $(date -u '+%F %T')"
  python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_s4_sd.yaml --out results/raw_test_s4 \
    --tf-lock $LOCK --worker-index 2 > logs/s4/sd_w1.log 2>&1 ) &
wait
echo "ITEM1_END $(date -u '+%F %T')"
