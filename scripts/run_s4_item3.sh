#!/bin/bash
# Session 4, item 3: cal-full = the S2+cal recipe (same traces, anchors, 1,415 steps x 16, lr 1e-5) in the
# full-context layout (micro-batches of 4 anchors sorted by length: same gradient), then a full-context replay of the
# 49 test traces with the calibrated weights (2 light workers; per-trace files, resumable).
source /home/rls_drafter_killtest/scripts/resume.sh >/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "CALFULL_TRAIN_START $(date -u '+%F %T')"
if [ ! -s ckpt/calfull/lr1e-05/model.safetensors ]; then
  python -m rlsdk.calibrate --traces traces/traces_cal.jsonl --lrs 1e-5 --steps 1415 --batch 16 --layout full \
    --micro ${MICRO:-4} --out ckpt/calfull > logs/s4/calfull_train.log 2>&1 || { echo "CALFULL TRAIN FAILED"; exit 1; }
fi
echo "CALFULL_TRAIN_END $(date -u '+%F %T')"
for w in 0 1; do
  python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_dev_baseline.yaml \
    --out results/raw_test_calfull --draft ckpt/calfull/lr1e-05 --tf-lock results/.s4_tf.lock --worker-index $((w * 2)) \
    > logs/s4/calfull_w$w.log 2>&1 &
done
wait
echo "CALFULL_RUN_END $(date -u '+%F %T')"
