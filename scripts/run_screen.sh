#!/bin/bash
# KL screen (HANDOFF NEXT SESSION step D): KL-bias × 3 and KL-linear × 3 on the 4 TTS-reference traces, fp32 adaptation.
#   bash scripts/run_screen.sh              real run on this GPU (A30, 24 GB) -> results/raw_a30, results/screen.md
#   DRY=1 bash scripts/run_screen.sh        end-to-end dry run on the tiny model (scratch dir; no commit/push/pause)
#   SCREEN=window bash scripts/run_screen.sh   windowed-context test instead (-> results/raw_a30_window, results/window.md)
#   SCREEN=mech bash scripts/run_screen.sh     mechanism + selection screen (-> results/raw_a30_window, results/mechanism.md)
#   SCREEN=farsel bash scripts/run_screen.sh   T0b target-guided far chunks (-> results/raw_a30_window, results/farsel.md)
# Env: WORKERS (concurrent pool workers, default 2 = fits 24 GB incl. the 10.7 GB teacher-forced transient),
# The instance is NOT paused by this script (decided 2026-10-10): it ends with a loud DONE; pause by hand.
# Resumable: finished (trace, method, config) files are skipped.
set -u
cd /home/rls_drafter_killtest
source scripts/resume.sh || { echo "RUN_SCREEN_ABORT resume"; exit 1; }
WORKERS=${WORKERS:-2}
# variable-size per-round tensors (context policies) fragment the caching allocator; this curbs it (numerics unchanged)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DRY=${DRY:-0}
SCREEN=${SCREEN:-kl}
REPORT_PY=scripts/screen_report.py
if [ "$SCREEN" = farsel ]; then
  KIND=farsel; RAW=results/raw_a30_window; MD=results/farsel.md; REPORT_PY=scripts/farsel_report.py
  ARMS="--row-cost results/attn_row_cost.md"
  MSG="T0b on A30: target-guided far-chunk selection S1-S3 + controls C1/C2, 4 TTS-reference traces (results/farsel.md)"
elif [ "$SCREEN" = mech ]; then
  KIND=mech; RAW=results/raw_a30_window; MD=results/mechanism.md; ARMS=""; REPORT_PY=scripts/mechanism_report.py
  MSG="Mechanism + selection screen on A30: B/E/C/C'/D context policies, 4 TTS-reference traces (results/mechanism.md)"
elif [ "$SCREEN" = window ]; then
  KIND=window; RAW=results/raw_a30_window; MD=results/window.md
  ARMS="--arms window --title 'Windowed drafter context (StreamingLLM-style) vs baseline (4 TTS-reference traces)' --stop-label 'close the long-output decay line'"
  MSG="Windowed-context test on A30: W x kept set, 4 TTS-reference traces (results/window.md)"
else
  KIND=screen; RAW=results/raw_a30; MD=results/screen.md; ARMS=""
  MSG="Screen on A30: KL-bias x3, KL-linear x3 on 4 TTS-reference traces (results/screen.md)"
fi
ts() { date -u +%FT%TZ; }
IDS="aime24_II_14 aime24_I_10 lcb_abc393_d lcb_arc191_d"

if [ "$DRY" = 1 ]; then
  WORK=${DRY_DIR:-/tmp/run_screen_dry}; rm -rf $WORK; mkdir -p $WORK
  $PY scripts/make_plans.py $KIND --out $WORK/plan.yaml || exit 1
  $PY -c "
import sys; sys.path.insert(0, 'tests')
from test_resume import _write_traces
import json
_write_traces('$WORK/traces.jsonl', n=4, new=300)
rows = [json.loads(l) for l in open('$WORK/traces.jsonl')]
for r, tid, ds in zip(rows, '$IDS'.split(), ['aime', 'aime', 'lcb', 'lcb']):
    r.update(trace_id=tid, dataset=ds, loop=False)
open('$WORK/traces.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
" || exit 1
  TRACES=$WORK/traces.jsonl PLAN=$WORK/plan.yaml OUT=$WORK/raw LOGS=$WORK/logs EXTRA="--tiny --max-new-tokens 300"
  REPORT="--raw $WORK/raw --cuts /nonexistent --tts-gain 1.0 --out $WORK/report.md --gpu tiny"
else
  $PY scripts/make_plans.py $KIND --out configs/plan_$KIND.yaml || exit 1
  TRACES=traces/traces.jsonl PLAN=configs/plan_$KIND.yaml OUT=$RAW LOGS=logs/$KIND EXTRA=""
  REPORT="--raw $RAW --out $MD"
fi
mkdir -p $LOGS
echo "SCREEN_START $(ts) workers=$WORKERS dry=$DRY"
pids=()
for i in $(seq 0 $((WORKERS - 1))); do
  $PY -m rlsdk.pool --traces $TRACES --split dev --ids $IDS --plan $PLAN --out $OUT --kind light \
      --tf-lock $OUT.tf.lock --worker-index $i $EXTRA > $LOGS/screen_w$i.log 2>&1 &
  pids+=($!); [ "$DRY" = 1 ] || sleep 20   # stagger model loads
done
rc=0; for p in ${pids[@]}; do wait $p || rc=$?; done
echo "SCREEN_POOL_END $(ts) rc=$rc"
grep -h "tau=" $LOGS/screen_w*.log
if [ $rc != 0 ] || grep -q "HARD STOP\|Traceback\|OutOfMemory" $LOGS/screen_w*.log; then
  echo "RUN_SCREEN_ABORT pool (see $LOGS)"; grep -h -A3 "Traceback\|HARD STOP\|OutOfMemory" $LOGS/screen_w*.log | tail -20
  FAILED=1
else
  eval $PY $REPORT_PY $REPORT $ARMS || FAILED=1
fi

if [ "$DRY" = 1 ]; then
  echo "DRY: would run: git add $MD configs/plan_$KIND.yaml $RAW/*.jsonl && git commit && git push"
  git push --dry-run origin HEAD 2>&1 | tail -2
else
  git add $MD configs/plan_$KIND.yaml $RAW/*.jsonl
  git commit -q -m "$MSG" \
     -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "COMMITTED $(git rev-parse --short HEAD)"
  git push -q origin HEAD && echo "PUSHED" || echo "PUSH FAILED (commit is local)"
fi

echo "SCREEN_END $(ts) failed=${FAILED:-0}"
echo "################################################################"
echo "###  DONE — screen finished. PAUSE THE INSTANCE BY HAND NOW.  ###"
echo "################################################################"
