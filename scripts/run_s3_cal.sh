#!/bin/bash
# Session 3, Steps 2-3 (+ the lr-pick evaluations and Step 4's TTS stride 1 runs). Pre-registered in RUNLOG.md.
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REF="aime24_II_14 aime24_I_10 lcb_abc393_d lcb_arc191_d"
LOCK=results/.s3_tf.lock
until grep -q "^END" logs/s3/step1.log; do sleep 20; done
echo "STEP2_START $(date -u '+%F %T')"
$ENV_VLLM/bin/python scripts/gen_traces.py --set cal --out traces/traces_cal.jsonl --max-new-tokens 8192 || { echo "GEN FAILED"; exit 1; }
python - <<'PY'
import json, csv
from rlsdk.data import loop_cut
rows=[json.loads(l) for l in open("traces/traces_cal.jsonl")]
with open("results/loop_cuts_cal.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["trace_id","dataset","n_out","finish_reason","loop","c","dropped"])
    for r in rows:
        c=loop_cut(r["output_ids"]) if r["loop"] else None
        w.writerow([r["trace_id"],r["dataset"],len(r["output_ids"]),r["finish_reason"],r["loop"],
                    c["c"] if c else "", c["dropped"] if c else False])
n=[len(r["output_ids"]) for r in rows]
print(f"CAL TRACES {len(rows)}: loop-flagged {sum(r['loop'] for r in rows)}, hit cap {sum(r['finish_reason']=='length' for r in rows)}, "
      f"output tokens total {sum(n)}, median {sorted(n)[len(n)//2]}, min {min(n)}, max {max(n)}")
PY
echo "STEP2_END $(date -u '+%F %T')"
echo "STEP3_START $(date -u '+%F %T')"
python -m rlsdk.calibrate --traces traces/traces_cal.jsonl --lrs 3e-6 1e-5 --budget-h 0.75 --batch 16 --out ckpt/cal \
  || { echo "CALIBRATE FAILED"; exit 1; }
echo "STEP3_END $(date -u '+%F %T')"
echo "EVAL_START $(date -u '+%F %T')"
# 2 processes at a time on 24 GB: the two S2+cal evals in sequence, next to one TTS worker
(for lr in 3e-06 1e-05; do
  python -m rlsdk.pool --traces traces/traces.jsonl --ids $REF --plan configs/plan_s3_s2.yaml \
    --out results/raw_a30_cal_lr$lr --draft ckpt/cal/lr$lr --tf-lock $LOCK > logs/s3/eval_lr$lr.log 2>&1
done) &
python -m rlsdk.pool --traces traces/traces.jsonl --ids $REF --plan configs/plan_s3_tts1.yaml \
  --out results/raw_a30_s3 --tf-lock $LOCK > logs/s3/tts1_ref.log 2>&1 &
wait
echo "EVAL_END $(date -u '+%F %T')"
python scripts/s3_report.py lrpick
