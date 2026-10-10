#!/bin/bash
# VPA follow-on (pre-registered addendum B, RUNLOG.md). Waits for the held-out driver to commit + push (GRACE), takes
# over the pause, merges branch vpa into the main checkout, then:
#   step 1  screen M = 64 x 7 (beta, lam) on the 4 reference traces -> pick the best by mean τ
#   step 2  confirm window, S2, chosen VPA on the 7 remaining non-dropped dev traces -> promotion rule
#   freeze  configs/frozen.yaml committed + pushed before any test run (only if promoted)
#   T3      VPA on all 50 test traces (only if promoted and within the 2 A30-h VPA budget)
#   report  results/vpa.md, VPA section in summary.md after the held-out section, RUNLOG row, commit, push,
#           grace (logs/vpa/HOLD keeps the instance up), pause via the JarvisLabs API.
# Runs only if the held-out run completed; otherwise stands down (and pauses if the held-out driver died unpaused).
set -u
MAIN=/home/rls_drafter_killtest
cd $MAIN
LOGS=$MAIN/logs/vpa; mkdir -p $LOGS
HLOG=$MAIN/logs/heldout/driver.log
CAP_H=${CAP_H:-2}
GRACE_S=${GRACE_S:-600}
ts() { date -u +%FT%TZ; }
pause_now() {
  echo "PAUSING $(ts)"
  for k in 1 2 3; do /home/envs/env-jl/bin/python scripts/jl_pause.py --pause && exit 0; sleep 60; done
  echo "###  PAUSE VIA API FAILED — PAUSE THE INSTANCE BY HAND NOW.  ###"; exit 1
}

echo "VPA_WAIT $(ts)"
while true; do
  if grep -q "^GRACE" $HLOG 2>/dev/null; then break; fi
  if ! pgrep -f "scripts/run_heldout.sh" >/dev/null; then
    sleep 5; grep -q "^GRACE\|^PAUSING" $HLOG && break
    echo "VPA_STANDDOWN held-out driver gone without GRACE $(ts)"; pause_now
  fi
  sleep 30
done
if ! grep -q "^FINISH .*reason=complete" $HLOG; then
  echo "VPA_STANDDOWN held-out did not complete ($(grep '^FINISH' $HLOG)) — leaving the held-out pause in place"; exit 0
fi
touch $MAIN/logs/heldout/HOLD
pkill -f "scripts/run_heldout.sh"; sleep 2
echo "VPA_TAKEOVER $(ts) (held-out committed + pushed; its driver stopped during grace; this driver now owns the pause)"
START=$(date +%s)
elapsed_h() { awk -v s=$START -v n=$(date +%s) 'BEGIN { printf "%.3f", (n - s) / 3600 }'; }
fits() { awk -v e=$(elapsed_h) -v x=$1 -v c=$CAP_H 'BEGIN { exit !(e + x <= c) }'; }

git merge --no-edit vpa > $LOGS/merge.log 2>&1 || { git merge --abort; echo "VPA_MERGE_FAILED"; cat $LOGS/merge.log; pause_now; }
echo "MERGED $(git rev-parse --short HEAD)"
source scripts/resume.sh || { echo "VPA_ABORT resume"; pause_now; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
REF="aime24_II_14 aime24_I_10 lcb_abc393_d lcb_arc191_d"
CONF="aime24_II_02 aime24_II_09 aime24_II_13 aime24_I_13 lcb_3701 lcb_arc193_d lcb_arc195_c"

phase() {  # phase NAME N PLAN SPLIT OUT IDS...
  local name=$1 n=$2 plan=$3 split=$4 out=$5; shift 5
  for attempt in 1 2; do
    echo "PHASE_START $name attempt=$attempt $(ts) elapsed_h=$(elapsed_h)"
    local pids=() rc=0
    for i in $(seq 0 $((n-1))); do
      $PY -m rlsdk.pool --traces traces/traces.jsonl --split $split --ids "$@" --plan $plan --out $out --kind light \
          --tf-lock $out.tf.lock --worker-index $i > $LOGS/${name}_a${attempt}_w$i.log 2>&1 &
      pids+=($!); sleep 20
    done
    for p in ${pids[@]}; do wait $p || rc=$?; done
    echo "PHASE_END $name attempt=$attempt rc=$rc $(ts) elapsed_h=$(elapsed_h)"
    grep -h "HARD STOP" $LOGS/${name}_a${attempt}_w*.log && return 3
    [ $rc = 0 ] && ! grep -q "Traceback\|OutOfMemory" $LOGS/${name}_a${attempt}_w*.log && return 0
    n=1
  done
  return 1
}

finish() {
  echo "VPA_FINISH $(ts) elapsed_h=$(elapsed_h) reason=$1"
  $PY scripts/vpa_report.py --stage "$1" || echo "VPA_REPORT_FAILED"
  E=$(date -u +%H:%M:%S); S0=$(date -u -d @$START +%H:%M:%S)
  printf '\n| # | start (UTC) | end (UTC) | command | A30-h | note |\n|---|---|---|---|---|---|\n| 33 | 2026-10-10 %s | %s (before grace + pause) | `bash scripts/run_vpa.sh` | %s | %s |\n' \
      "$S0" "$E" "$(elapsed_h)" "$1" >> RUNLOG.md
  git add results/vpa.md results/summary.md results/raw_a30_window/*vpa*.jsonl results/raw_a30_dev/*.jsonl \
      results/raw_test/*vpa*.jsonl configs/plan_vpa_*.yaml RUNLOG.md 2>/dev/null
  git commit -q -m "VPA (addendum B, $1): results/vpa.md, summary.md" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "COMMITTED $(git rev-parse --short HEAD)"
  git push -q origin HEAD && echo "PUSHED" || echo "PUSH FAILED (commit is local)"
  echo "GRACE ${GRACE_S}s (touch $LOGS/HOLD to keep the instance up) $(ts)"
  sleep $GRACE_S
  while [ -f $LOGS/HOLD ] && [ $(( $(date +%s) - START )) -lt $(( (CAP_H + 1) * 3600 )) ]; do sleep 30; done
  git add -A results/vpa.md results/summary.md RUNLOG.md 2>/dev/null
  git commit -q -m "VPA: final edits before pause" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
      && git push -q origin HEAD
  pause_now
}

# ---- step 1: screen on the 4 reference traces
cat > configs/plan_vpa_screen.yaml <<EOF
- {method: vpa, config: {M: 64, beta: 0.0, lam: 0.0}}
- {method: vpa, config: {M: 64, beta: 0.5, lam: 0.0}}
- {method: vpa, config: {M: 64, beta: 0.5, lam: 0.5}}
- {method: vpa, config: {M: 64, beta: 0.5, lam: 1.0}}
- {method: vpa, config: {M: 64, beta: 1.0, lam: 0.0}}
- {method: vpa, config: {M: 64, beta: 1.0, lam: 0.5}}
- {method: vpa, config: {M: 64, beta: 1.0, lam: 1.0}}
EOF
fits 0.6 || finish "budget before step 1"
phase vpa_screen 2 configs/plan_vpa_screen.yaml dev results/raw_a30_window $REF || finish "step 1 failed"
BEST=$($PY scripts/vpa_report.py --pick) || finish "pick failed"
echo "VPA_BEST $BEST"

# ---- step 2: confirm on the 7 remaining non-dropped dev traces
$PY - "$BEST" <<'EOF'
import sys, yaml
best = dict(kv.split("=") for kv in sys.argv[1].split("_"))
cfg = {"M": int(best["M"]), "beta": float(best["beta"]), "lam": float(best["lam"])}
plan = [{"method": "window", "config": {"W": 1024, "keep": "prompt"}},
        {"method": "farsel", "config": {"W": 1024, "keep": "prompt", "n": 16, "score": "mid3"}},
        {"method": "vpa", "config": cfg}]
yaml.safe_dump(plan, open("configs/plan_vpa_confirm.yaml", "w"), sort_keys=False)
EOF
fits 0.5 || finish "budget before step 2"
phase vpa_confirm 2 configs/plan_vpa_confirm.yaml dev results/raw_a30_dev $CONF || finish "step 2 failed"
PROMO=$($PY scripts/vpa_report.py --promote "$BEST")
echo "VPA_PROMOTION $PROMO"
case "$PROMO" in
  PROMOTE*) ;;
  *) finish "not promoted";;
esac

# ---- freeze before any test run
$PY - "$BEST" <<'EOF'
import sys, yaml
best = dict(kv.split("=") for kv in sys.argv[1].split("_"))
frozen = {"note": "Frozen 2026-10-10 before T3 (addendum B). TTS setting from T0.",
          "plan": [{"method": "baseline"},
                   {"method": "window", "config": {"W": 1024, "keep": "prompt"}},
                   {"method": "farsel", "config": {"W": 1024, "keep": "prompt", "n": 16, "score": "mid3"}},
                   {"method": "vpa", "config": {"M": int(best["M"]), "beta": float(best["beta"]), "lam": float(best["lam"])}},
                   {"method": "tts", "config": {"lr": 1.0e-05}}]}
yaml.safe_dump(frozen, open("configs/frozen.yaml", "w"), sort_keys=False)
yaml.safe_dump([frozen["plan"][3]], open("configs/plan_vpa_t3.yaml", "w"), sort_keys=False)
EOF
git add configs/frozen.yaml configs/plan_vpa_t3.yaml configs/plan_vpa_screen.yaml configs/plan_vpa_confirm.yaml \
    results/raw_a30_window/*vpa*.jsonl results/raw_a30_dev/*.jsonl
git commit -q -m "Freeze VPA config ($BEST) in configs/frozen.yaml before T3" \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD && echo "FROZEN_PUSHED $(git rev-parse --short HEAD)"

# ---- T3: VPA on all test traces
fits 0.7 || finish "budget: T3 projection would exceed $CAP_H A30-h"
TEST=$($PY -c "import json; print(' '.join(json.loads(l)['trace_id'] for l in open('traces/traces.jsonl') if json.loads(l)['split'] == 'test'))")
phase vpa_t3 2 configs/plan_vpa_t3.yaml test results/raw_test $TEST || finish "T3 failed"
finish "complete"
