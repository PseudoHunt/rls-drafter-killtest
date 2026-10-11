#!/bin/bash
# One command: rerun S2+cal (static, frozen calibrated drafter) on all test traces and compare with the session-3 result.
#   bash scripts/rerun_s2cal.sh            (A30: ≈ 0.4 A30-h; resumable, finished traces are skipped)
source /home/rls_drafter_killtest/scripts/resume.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=ckpt/cal/lr1e-05; WANT=cc25cc15a3782b56e2a5be9d1a17bbaf74718be966413447b76a5b728e8ca5d7
OUT=${OUT:-results/raw_test_cal_rerun}
GOT=$(python -c "from rlsdk.calibrate import sha256_dir; print(sha256_dir('$CKPT'))") || exit 1
[ "$GOT" = "$WANT" ] || { echo "CHECKPOINT HASH MISMATCH: $GOT (want $WANT). Rebuild: see HANDOFF.md"; exit 1; }
echo "checkpoint ok ($WANT)"
python -m rlsdk.pool --traces traces/traces.jsonl --split test --plan configs/plan_s3_s2.yaml --out $OUT --draft $CKPT || exit 1
python - "$OUT" <<'PY'
import json, sys
sys.argv = [sys.argv[0], sys.argv[1]]; sys.path.insert(0, "scripts")
import s3_report as s
ids = sorted(json.loads(x)["trace_id"] for x in open(s.R("traces/traces.jsonl")) if json.loads(x)["split"] == "test")
a = s.Arm("S2+cal rerun", s.R(sys.argv[1]), s.S2, ids, s.cuts())
t = s.tau(a.stats(), sorted(a.dec))
print(f"S2+cal τ (decision variant, {len(a.dec)} traces) = {t:.3f}; session-3 result 4.408 "
      f"({'MATCH' if abs(t - 4.408) < 0.0015 else 'DIFFERS: same GPU type? (bf16 numerics differ across GPUs)'})")
PY
