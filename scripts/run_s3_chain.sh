#!/bin/bash
# Session 3 chain: after Step 6 -> Step 1 rerun (stride 10, micro 10 + gqa_attention) -> τ guard -> lr pick -> Step 4.
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "^STEP6_END" logs/s3/step6.log; do sleep 15; done
echo "STEP1C_START $(date -u '+%F %T')"
python -m rlsdk.pool --traces traces/traces.jsonl --ids $(cat configs/t2_traces.txt) \
  --plan configs/plan_s3_batched.yaml --out results/raw_test --kind heavy || { echo "STEP1C FAILED"; exit 1; }
echo "STEP1C_END $(date -u '+%F %T')"
python scripts/s3_report.py step1 > logs/s3/step1_report.txt 2>&1
grep -E "τ check|cost criterion" logs/s3/step1_report.txt
grep -q "τ check.*PASS" logs/s3/step1_report.txt || { echo "STEP1 TAU CHECK FAILED - stopping before Step 4"; exit 1; }
python scripts/s3_report.py lrpick | tee logs/s3/lrpick.txt
LR=$(python - <<'PY'
import re
best=max((float(m.group(2)), m.group(1)) for m in (re.match(r"raw_a30_cal_lr(\S+): S2\+cal τ = ([0-9.]+)", l) for l in open("logs/s3/lrpick.txt")) if m)
print(best[1])
PY
)
echo "LR_PICKED $LR"
bash scripts/run_s3_screen.sh $LR
