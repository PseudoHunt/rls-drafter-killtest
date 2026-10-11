#!/bin/bash
# Session 3, Step 5: promoted arms on all test traces. Usage: run_s3_heldout.sh <lr> <arms...>  (arms: S2+cal full warm)
# One process per drafter, jobs sequential (each arm timed alone on the GPU); S2+cal and full share the calibrated drafter.
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LR=$1; shift
echo "STEP5_START $(date -u '+%F %T') lr=$LR arms=$*"
PLAN=logs/s3/plan_heldout_cal.yaml; : > $PLAN
for a in "$@"; do
  [ "$a" = "S2+cal" ] && cat configs/plan_s3_s2.yaml >> $PLAN
  [ "$a" = "full" ] && cat configs/plan_s3_full.yaml >> $PLAN
done
if [ -s $PLAN ]; then
  python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan $PLAN --out results/raw_test_cal \
    --draft ckpt/cal/lr$LR || { echo "STEP5 FAILED"; exit 1; }
fi
if echo "$*" | grep -qw warm; then
  python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_s3_warm.yaml \
    --out results/raw_test_s3 || { echo "STEP5 WARM FAILED"; exit 1; }
fi
echo "STEP5_END $(date -u '+%F %T')"
python scripts/s3_report.py heldout --arms "$@"
