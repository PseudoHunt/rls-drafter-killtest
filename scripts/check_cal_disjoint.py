"""Session 3: assert the calibration set is disjoint from the 62 dev/test problems, and the 4 reference traces (used to
pick the calibration lr) are not test traces. Output -> results/cal_disjoint.txt.

  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. $ENV_VLLM/bin/python scripts/check_cal_disjoint.py
"""
import json
import re

from rlsdk.data import CAL_EXCLUDE, load_calibration_problems, load_problems

REF = ["aime24_II_14", "aime24_I_10", "lcb_abc393_d", "lcb_arc191_d"]


def body(s):
    return re.sub(r"\s+", " ", s.split("### Question:")[-1].lower())


def shingles(s, n=8):
    w = re.findall(r"\w+", body(s))
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    cal, main_ = load_calibration_problems(), load_problems()
    out = []
    log = lambda m: (print(m), out.append(m))  # noqa: E731
    log(f"calibration problems: {len(cal)} (AIME {sum(c['dataset'] == 'aime' for c in cal)}, "
        f"LCB {sum(c['dataset'] == 'lcb' for c in cal)}); dev/test problems: {len(main_)}")
    ids = {m["trace_id"] for m in main_} & {c["trace_id"] for c in cal}
    assert not ids, ids
    log("assert no shared trace ids: OK (0)")
    full = {m["prompt"] for m in main_} & {c["prompt"] for c in cal}
    assert not full
    log("assert no identical prompts: OK (0)")
    pre = {body(m["prompt"])[:300] for m in main_} & {body(c["prompt"])[:300] for c in cal}
    assert not pre, pre
    log("assert no shared first 300 chars of the problem statement: OK (0)")
    ms = [(m["trace_id"], shingles(m["prompt"])) for m in main_]
    worst = max((len(a & b) / max(1, len(a | b)), c["trace_id"], mid)
                for c in cal for a in [shingles(c["prompt"])] for mid, b in ms)
    assert worst[0] < 0.5, worst
    log(f"assert max 8-word-shingle Jaccard (statement) < 0.5: OK (max {worst[0]:.3f}, {worst[1]} vs {worst[2]})")
    assert not any(c["trace_id"][4:] in CAL_EXCLUDE for c in cal)
    log(f"excluded near-duplicate(s) not in the set: {sorted(CAL_EXCLUDE)} (abc388_g ~ test abc388_e)")
    test = {m["trace_id"] for m in main_ if m["split"] == "test"}
    dev = {m["trace_id"] for m in main_ if m["split"] == "dev"}
    assert not set(REF) & test and set(REF) <= dev
    log(f"assert the 4 reference traces are dev, not test: OK ({', '.join(REF)})")
    traces = [json.loads(x)["trace_id"] for x in open("traces/traces_cal.jsonl")]
    assert sorted(traces) == sorted(c["trace_id"] for c in cal)
    log(f"assert traces/traces_cal.jsonl holds exactly these {len(traces)} problems: OK")
    log("ALL DISJOINTNESS ASSERTIONS PASSED")
    open("results/cal_disjoint.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
