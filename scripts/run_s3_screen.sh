#!/bin/bash
# Session 3, Step 4 screen (+ Step 5 held-out if a promotion rule triggers). Usage: run_s3_screen.sh <lr, e.g. 1e-05>
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LR=$1; CAL=ckpt/cal/lr$LR; CALDIR=results/raw_a30_cal_lr$LR
REF="aime24_II_14 aime24_I_10 lcb_abc393_d lcb_arc191_d"
LOCK=results/.s3_tf.lock
echo "STEP4_START $(date -u '+%F %T') lr=$LR"
python -m rlsdk.pool --traces traces/traces.jsonl --ids $REF --plan configs/plan_s3_full.yaml --out $CALDIR \
  --draft $CAL --tf-lock $LOCK || { echo "FULL FAILED"; exit 1; }
python scripts/s3_report.py screen --cal-dir $CALDIR > logs/s3/screen_report.txt
R=$(grep '^READINGS' logs/s3/screen_report.txt | cut -d' ' -f2-)
echo "READINGS1 $R"
if echo "$R" | grep -q '"mode_mismatch": false'; then
  echo "WARM_ARM_START $(date -u '+%F %T')"
  python -m rlsdk.pool --traces traces/traces.jsonl --ids $REF --plan configs/plan_s3_warm.yaml --out results/raw_a30_s3 \
    --tf-lock $LOCK || { echo "WARM FAILED"; exit 1; }
  python scripts/s3_report.py screen --cal-dir $CALDIR > logs/s3/screen_report.txt
  R=$(grep '^READINGS' logs/s3/screen_report.txt | cut -d' ' -f2-)
  echo "READINGS2 $R"
fi
echo "STEP4_END $(date -u '+%F %T')"
