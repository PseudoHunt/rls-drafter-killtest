#!/bin/bash
# Held-out test (pre-registered in RUNLOG.md, 2026-10-10). End to end, unattended:
#   T1a  all 50 test traces, light arms: baseline, window (W 1024 + prompt), S2, C1 (window W 1536), C2 (random chunks)
#   gate τ_window − τ_base ≥ 0.4 on test (decision variant) — else report, commit, push, pause
#   T1b  TTS stride 1 (TTS setting: lr 1e-5, stale) on all 50 test traces
#   T2a  16 test traces: TTS stride 10; S2 + TTS stride 10 pooled
#   T2b  16 test traces: S2 + TTS stride 1 (first to be cut if the 9 A30-h projection is exceeded)
#   isolated drafting timing (baseline, window, S2; 2 dev traces, one process alone), report, summary, commit, push,
#   grace period (logs/heldout/HOLD keeps the instance up), pause via the JarvisLabs API.
#   DRY=1: tiny model, scratch dir, no commit/push/pause (API check only).
set -u
cd /home/rls_drafter_killtest
source scripts/resume.sh || { echo "HELDOUT_ABORT resume"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DRY=${DRY:-0}
CAP_H=${CAP_H:-9}
GRACE_S=${GRACE_S:-600}
LIGHT_WORKERS=${LIGHT_WORKERS:-2}
START=$(date +%s)
ts() { date -u +%FT%TZ; }
elapsed_h() { awk -v s=$START -v n=$(date +%s) 'BEGIN { printf "%.3f", (n - s) / 3600 }'; }
LR=1.0e-05  # T0: best stale lr on the AIME dev grid

if [ "$DRY" = 1 ]; then
  WORK=${DRY_DIR:-/tmp/run_heldout_dry}; rm -rf $WORK; mkdir -p $WORK
  $PY -c "
import sys, json; sys.path.insert(0, 'tests')
from test_resume import _write_traces
_write_traces('$WORK/traces.jsonl', n=6, new=300)
rows = [json.loads(l) for l in open('$WORK/traces.jsonl')]
for i, r in enumerate(rows):
    r.update(trace_id=f'tiny{i}', dataset='aime' if i < 3 else 'lcb', split='test', loop=False)
open('$WORK/traces.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
open('$WORK/t2_traces.txt', 'w').write('tiny0 tiny1 tiny3 tiny4\n')
" || exit 1
  TRACES=$WORK/traces.jsonl OUT=$WORK/raw LOGS=$WORK/logs CFG=$WORK EXTRA="--tiny --max-new-tokens 300"
  T2IDS=$(cat $WORK/t2_traces.txt); ISO_IDS="tiny0 tiny3"; ISO_SPLIT=test; ISO_TRACES=$TRACES; ISO_OUT=$WORK/iso
  REPORT_ARGS="--raw $OUT --iso $ISO_OUT --cuts /nonexistent --t2-ids $T2IDS --out $WORK/heldout.md --summary $WORK/summary.md --iters 200"
else
  TRACES=traces/traces.jsonl OUT=results/raw_test LOGS=logs/heldout CFG=configs EXTRA=""
  T2IDS=$(cat configs/t2_traces.txt); ISO_IDS="aime24_II_14 lcb_abc393_d"; ISO_SPLIT=dev; ISO_TRACES=$TRACES
  ISO_OUT=results/timing_isolated_a30
  REPORT_ARGS="--raw $OUT --iso $ISO_OUT --t2-ids $T2IDS"
fi
mkdir -p $LOGS $OUT

S2='{W: 1024, keep: prompt, n: 16, score: mid3}'
cat > $CFG/plan_t1_light.yaml <<EOF
- {method: baseline}
- {method: window, config: {W: 1024, keep: prompt}}
- {method: farsel, config: $S2}
- {method: window, config: {W: 1536, keep: prompt}}
- {method: farsel, config: {W: 1024, keep: prompt, n: 16, score: random, seed: 0}}
EOF
cat > $CFG/plan_t1_tts.yaml <<EOF
- {method: tts, config: {lr: $LR}}
EOF
cat > $CFG/plan_t2a.yaml <<EOF
- {method: tts, config: {lr: $LR, stride: 10}}
- {method: tts, config: {lr: $LR, ctx: s2, stride: 10, pool: true}}
EOF
cat > $CFG/plan_t2b.yaml <<EOF
- {method: tts, config: {lr: $LR, ctx: s2}}
EOF
cat > $CFG/plan_iso.yaml <<EOF
- {method: baseline}
- {method: window, config: {W: 1024, keep: prompt}}
- {method: farsel, config: $S2}
EOF

phase() {  # phase NAME N_WORKERS KIND PLAN SPLIT TRACES OUTDIR [IDS...]
  local name=$1 n=$2 kind=$3 plan=$4 split=$5 traces=$6 out=$7; shift 7
  local ids=() pids=() rc=0; [ $# -gt 0 ] && ids=(--ids "$@")
  for attempt in 1 2; do
    echo "PHASE_START $name attempt=$attempt workers=$n $(ts) elapsed_h=$(elapsed_h)"
    pids=(); rc=0
    for i in $(seq 0 $((n-1))); do
      $PY -m rlsdk.pool --traces $traces --split $split "${ids[@]}" --plan $plan --out $out --kind $kind \
          --tf-lock $out.tf.lock --worker-index $i $EXTRA > $LOGS/${name}_a${attempt}_w$i.log 2>&1 &
      pids+=($!); [ "$DRY" = 1 ] || sleep 20
    done
    for p in ${pids[@]}; do wait $p || rc=$?; done
    echo "PHASE_END $name attempt=$attempt rc=$rc $(ts) elapsed_h=$(elapsed_h)"
    grep -h "HARD STOP" $LOGS/${name}_a${attempt}_w*.log && { echo "HELDOUT_HARD_STOP $name"; return 3; }
    if [ $rc = 0 ] && ! grep -q "Traceback\|OutOfMemory" $LOGS/${name}_a${attempt}_w*.log; then return 0; fi
    grep -h -A2 "Error" $LOGS/${name}_a${attempt}_w*.log | tail -5
    n=1  # resumable: retry once with a single worker
  done
  echo "HELDOUT_PHASE_FAILED $name"; return 1
}

report() {
  $PY scripts/heldout_report.py $REPORT_ARGS "$@" || echo "HELDOUT_REPORT_FAILED"
}

finish() {  # commit, push, grace period, pause
  echo "FINISH $(ts) elapsed_h=$(elapsed_h) reason=$1"
  if [ "$DRY" = 1 ]; then
    echo "DRY: would commit and push results/heldout.md results/summary.md $OUT"
    /home/envs/env-jl/bin/python scripts/jl_pause.py --check || echo "JL check failed"
    echo "HELDOUT_DONE (dry)"; exit 0
  fi
  E=$(date -u +%H:%M:%S); S0=$(date -u -d @$START +%H:%M:%S)
  printf '\n| # | start (UTC) | end (UTC) | command | A30-h | note |\n|---|---|---|---|---|---|\n| 32 | 2026-10-10 %s | %s (before grace + pause) | `bash scripts/run_heldout.sh` | %s | %s; phases: %s |\n' \
      "$S0" "$E" "$(elapsed_h)" "$1" "$(grep -h '^PHASE_END\|^CUT' $LOGS/driver.log 2>/dev/null | awk '{print $2, $4, $NF}' | tr '\n' ';')" >> RUNLOG.md
  git add results/heldout.md results/summary.md results/raw_test/*.jsonl results/timing_isolated_a30/*.jsonl \
      configs/plan_t1_*.yaml configs/plan_t2*.yaml configs/plan_iso.yaml configs/t2_traces.txt RUNLOG.md results/loop_check_proposal.md 2>/dev/null
  git commit -q -m "Held-out test on A30 ($1): results/heldout.md, summary.md" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "COMMITTED $(git rev-parse --short HEAD)"
  git push -q origin HEAD && echo "PUSHED" || echo "PUSH FAILED (commit is local)"
  echo "GRACE ${GRACE_S}s (touch $LOGS/HOLD to keep the instance up; remove it to continue) $(ts)"
  sleep $GRACE_S
  while [ -f $LOGS/HOLD ] && [ $(( $(date +%s) - START )) -lt $(( (CAP_H + 1) * 3600 )) ]; do sleep 30; done
  git add -A results/heldout.md results/summary.md RUNLOG.md 2>/dev/null
  git commit -q -m "Held-out: final edits before pause" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
      && git push -q origin HEAD
  echo "PAUSING $(ts) elapsed_h=$(elapsed_h)"
  for k in 1 2 3; do /home/envs/env-jl/bin/python scripts/jl_pause.py --pause && exit 0; sleep 60; done
  echo "################################################################"
  echo "###  PAUSE VIA API FAILED — PAUSE THE INSTANCE BY HAND NOW.  ###"
  echo "################################################################"
  exit 1
}

echo "HELDOUT_START $(ts) cap_h=$CAP_H dry=$DRY t2=[$T2IDS]"

# ---- T1a: light arms on all test traces
phase t1_light $LIGHT_WORKERS light $CFG/plan_t1_light.yaml test $TRACES $OUT; rc=$?
[ $rc = 0 ] || { report --stage t1a_failed; finish "T1a failed rc=$rc"; }

# ---- gate
GATE=$($PY scripts/heldout_report.py $REPORT_ARGS --gate-only)
[ "$DRY" = 1 ] && [ "${DRY_FORCE_GATE:-0}" = 1 ] && GATE="PASS (forced, dry run only) $GATE"
echo "GATE $GATE"
case "$GATE" in
  PASS*) ;;
  *) report --stage gate_stop; finish "gate: window - base < 0.4";;
esac

# ---- T1b: TTS stride 1 on all test traces (one TTS worker fits on 24 GB)
phase t1_tts 1 heavy $CFG/plan_t1_tts.yaml test $TRACES $OUT; rc=$?
[ $rc = 0 ] || { report --stage t1b_failed; finish "T1b failed rc=$rc"; }

# ---- T2a: TTS stride 10 + pooled hybrid on 16 traces (never cut)
phase t2a 1 heavy $CFG/plan_t2a.yaml test $TRACES $OUT $T2IDS; rc=$?
[ $rc = 0 ] || { report --stage t2a_failed; finish "T2a failed rc=$rc"; }

# ---- T2b: S2 + TTS stride 1 on 16 traces, if the 9 A30-h projection allows (first cut)
EST=$($PY scripts/heldout_report.py $REPORT_ARGS --estimate-t2b)  # hours, from measured S2-context TTS cost
FINAL_H=0.35  # isolated timing + report + grace
if awk -v e=$(elapsed_h) -v x=$EST -v f=$FINAL_H -v c=$CAP_H 'BEGIN { exit !(e + x + f <= c) }'; then
  phase t2b 1 heavy $CFG/plan_t2b.yaml test $TRACES $OUT $T2IDS || echo "T2b failed (reported as missing)"
else
  echo "CUT S2+TTS stride 1: elapsed $(elapsed_h) h + est $EST h + $FINAL_H h > $CAP_H h"
  echo "cut: S2 + TTS stride 1 (projection $(elapsed_h) + $EST + $FINAL_H h > $CAP_H h)" > $LOGS/CUT
fi

# ---- isolated drafting timing: one process alone on the GPU, 2 traces
phase iso 1 light $CFG/plan_iso.yaml $ISO_SPLIT $ISO_TRACES $ISO_OUT $ISO_IDS || echo "isolated timing failed"

report --stage done
finish "complete"
