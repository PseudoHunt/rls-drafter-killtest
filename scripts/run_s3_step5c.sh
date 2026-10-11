#!/bin/bash
# Session 3, Step 5b resume: "full" on the remaining test traces (resumable; up to 3 attempts, fresh process each).
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "STEP5C_START $(date -u '+%F %T')"
for i in 1 2 3; do
  python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_s3_full.yaml --out results/raw_test_cal \
    --draft ckpt/cal/lr1e-05 && break
  echo "ATTEMPT $i FAILED"; rm -f results/raw_test_cal/*.claim
done
n=$(ls results/raw_test_cal | grep -c "batched=True.*\.jsonl$")
echo "STEP5C_END $(date -u '+%F %T') full_files=$n"
