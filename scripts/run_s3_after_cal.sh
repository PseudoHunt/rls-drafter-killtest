#!/bin/bash
# Session 3: after the calibration driver: Step 1 rerun (micro-batched; stride 10 all 16 traces, stride 20 remaining),
# then the pre-registered lr pick (higher S2+cal τ on the 4 reference traces), then the Step 4 screen.
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "^EVAL_END" logs/s3/cal.log; do sleep 20; done
# S2+cal evals for the lr pick (rerun: the first attempt OOMed next to the TTS worker); 2 light workers, no TTS worker
REF="aime24_II_14 aime24_I_10 lcb_abc393_d lcb_arc191_d"
echo "EVAL2_START $(date -u '+%F %T')"
for lr in 3e-06 1e-05; do
  python -m rlsdk.pool --traces traces/traces.jsonl --ids $REF --plan configs/plan_s3_s2.yaml \
    --out results/raw_a30_cal_lr$lr --draft ckpt/cal/lr$lr --tf-lock results/.s3_tf.lock > logs/s3/eval2_lr$lr.log 2>&1 &
done
wait
for lr in 3e-06 1e-05; do [ $(ls results/raw_a30_cal_lr$lr/*.jsonl | wc -l) = 4 ] || { echo "EVAL2 FAILED lr=$lr"; exit 1; }; done
echo "EVAL2_END $(date -u '+%F %T')"
echo "STEP1B_START $(date -u '+%F %T')"
python -m rlsdk.pool --traces traces/traces.jsonl --ids $(cat configs/t2_traces.txt) \
  --plan configs/plan_s3_batched.yaml --out results/raw_test --kind heavy || { echo "STEP1B FAILED"; exit 1; }
echo "STEP1B_END $(date -u '+%F %T')"
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
