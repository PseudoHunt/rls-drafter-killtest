#!/bin/bash
# Session 4, item 1d: SharpDraft sanity anchor on the 4 reference traces (released drafter, full context), 2 workers.
source /home/rls_drafter_killtest/scripts/resume.sh >/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REF="aime24_II_14 aime24_I_10 lcb_abc393_d lcb_arc191_d"
echo "ANCHOR_START $(date -u '+%F %T')"
for w in 0 1; do
  python -m rlsdk.pool --traces traces/traces.jsonl --ids $REF --plan configs/plan_s4_sd.yaml --out results/raw_a30_s4 \
    --tf-lock results/.s4_tf.lock --worker-index $((w * 2)) > logs/s4/anchor_w$w.log 2>&1 &
done
wait
echo "ANCHOR_END $(date -u '+%F %T')"
