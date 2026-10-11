#!/bin/bash
# Session 3, Step 6: TTS stride 1 lr 1e-5 with full context-cache refresh after every step, 4 reference traces.
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "STEP6_START $(date -u '+%F %T')"
python -m rlsdk.pool --traces traces/traces.jsonl --ids aime24_II_14 aime24_I_10 lcb_abc393_d lcb_arc191_d \
  --plan configs/plan_s3_refresh.yaml --out results/raw_a30_s3 --tf-lock results/.s3_tf.lock || echo "STEP6 FAILED"
echo "STEP6_END $(date -u '+%F %T')"
