"""Small helpers called by scripts/stage3.sh."""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import load_cuts, load_raw, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sigma2():
    """sigma^2 from dev baseline runs on non-looping traces (z of all 15 slots, all rounds)."""
    runs = [r for r in load_raw(os.path.join(ROOT, "results/raw"), "dev")
            if r["header"]["method"] == "baseline" and not r["header"]["loop"]]
    s = sum(r["summary"]["z2_sum"] for r in runs)
    n = sum(r["summary"]["z2_count"] for r in runs)
    out = {"sigma2": s / n, "n_traces": len(runs), "n_values": n}
    json.dump(out, open(os.path.join(ROOT, "configs/sigma2.json"), "w"), indent=1)
    print(out["sigma2"])


def best_tts_lr():
    """Best stale TTS lr by dev mean tau (decision metric: truncation variant)."""
    tab = table(load_raw(os.path.join(ROOT, "results/raw"), "dev"), "trunc", load_cuts(os.path.join(ROOT, "results/loop_cuts.csv")))
    cands = {cfg: res["mean"]["tau"] for (m, cfg), res in tab.items() if m == "tts" and "refresh" not in cfg}
    best = max(cands, key=cands.get)
    print(best.split("=")[1])


def refresh_wins(lr):
    tab = table(load_raw(os.path.join(ROOT, "results/raw"), "dev"), "trunc", load_cuts(os.path.join(ROOT, "results/loop_cuts.csv")))
    stale = tab[("tts", f"lr={lr}")]["mean"]["tau"]
    ref = tab[("tts", f"lr={lr}_refresh=True")]["mean"]["tau"]
    print("1" if ref > 1.05 * stale else "0")


def use_fp64():
    """Amendment 1: switch RLS to fp64 if the check says so; rewrite the grid plan."""
    chk = json.load(open(os.path.join(ROOT, "results/rls_checks.json")))
    if not chk["lam1"]["pass"]:
        print("LAM1_FAIL")
        return
    if chk["fp64"]["switch_to_fp64"]:
        p = os.path.join(ROOT, "configs/plan_dev_grid.yaml")
        plan = yaml.safe_load(open(p))
        for item in plan:
            if item["method"] == "rls":
                item["config"]["dtype"] = "fp64"
        yaml.safe_dump(plan, open(p, "w"), sort_keys=False)
        print("FP64")
    else:
        print("FP32")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
