#!/bin/bash
# Session 4, item 2: thinking-mode drafter (pinned snapshot) baseline + S2 on the 49 test traces; 2 light workers,
# each runs both arms per trace (one teacher-forced pass per trace per worker).
source /home/rls_drafter_killtest/scripts/resume.sh >/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
THINK=hf_cache/hub/models--jgeuter--qwen3-4b-dpard-thinking-b16/snapshots/78baf299fc6dc793aab6c35ce44676a980e266d1
echo "ITEM2_START $(date -u '+%F %T')"
for w in 0 1; do
  python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_s4_think.yaml \
    --out results/raw_test_think --draft $THINK --tf-lock results/.s4_tf.lock --worker-index $((w * 2)) \
    > logs/s4/think_w$w.log 2>&1 &
done
wait
echo "ITEM2_END $(date -u '+%F %T')"
