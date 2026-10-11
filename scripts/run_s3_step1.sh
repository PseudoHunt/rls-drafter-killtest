#!/bin/bash
# Session 3, Step 1: batched pooled S2 + TTS (stride 10; stride 20 exploratory) on the 16 T2 traces; one TTS process
# alone on the GPU (trace-major: both strides share each trace's teacher-forced pass; jobs run sequentially).
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "START $(date -u '+%F %T')"
python -m rlsdk.pool --traces traces/traces.jsonl --ids $(cat configs/t2_traces.txt) \
  --plan configs/plan_s3_batched.yaml --out results/raw_test --kind heavy
echo "END $(date -u '+%F %T') rc=$?"
