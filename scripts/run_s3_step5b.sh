#!/bin/bash
# Session 3, Step 5b: "full" (S2+cal + batched TTS stride 10) on all test traces, after Step 5a (S2+cal), timed alone.
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "^STEP5A_END" logs/s3/step5a.log; do sleep 15; done
grep -q "STEP5A FAILED" logs/s3/step5a.log && { echo "STEP5A had failed; stopping"; exit 1; }
echo "STEP5B_START $(date -u '+%F %T')"
python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_s3_full.yaml --out results/raw_test_cal \
  --draft ckpt/cal/lr1e-05 || { echo "STEP5B FAILED"; exit 1; }
echo "STEP5B_END $(date -u '+%F %T')"
python scripts/s3_report.py heldout --arms S2+cal full --refresh-checked > logs/s3/heldout_report.txt 2>&1
echo "REPORT_DONE"
