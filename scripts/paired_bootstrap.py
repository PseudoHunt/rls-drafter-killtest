"""Paired bootstrap CI for τ differences between two arms on the same traces (pre-registered for P1/P3, RUNLOG.md).

Statistic: Δ = τ_A − τ_B, with τ = mean over datasets (AIME, LCB) of the pooled τ (committed tokens / rounds over the
dataset's traces), decision variant (flagged traces truncated at loop onset; dropped traces excluded).
Resampling: traces with replacement, stratified by dataset, the same draw for both arms (paired); B = 10,000;
seed 0; 95% percentile interval.

  python scripts/paired_bootstrap.py --raw DIR --a METHOD/CONFIG --b METHOD/CONFIG [--ids ...]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import DATASETS, load_cuts, load_raw, select  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def per_trace(runs, key, cuts):
    """{trace_id: (dataset, tokens, rounds)} for one (method, config), decision variant."""
    out = {}
    for run in select([r for r in runs if (r["header"]["method"], r["header"]["config"]) == key], "trunc", cuts):
        rs = run["rounds"]
        out[run["header"]["trace_id"]] = (run["header"]["dataset"], sum(r["tau"] for r in rs), len(rs))
    return out


def tau_mean(stats, ids):
    by = {ds: [0, 0] for ds in DATASETS}
    for i in ids:
        ds, tok, n = stats[i]
        by[ds][0] += tok
        by[ds][1] += n
    return sum(by[ds][0] / by[ds][1] for ds in DATASETS) / len(DATASETS)


def paired_bootstrap(A, B, B_iter=10000, seed=0):
    ids = sorted(set(A) & set(B))
    strata = {ds: [i for i in ids if A[i][0] == ds] for ds in DATASETS}
    point = tau_mean(A, ids) - tau_mean(B, ids)
    rng = random.Random(seed)
    diffs = []
    for _ in range(B_iter):
        draw = [rng.choice(strata[ds]) for ds in DATASETS for _ in strata[ds]]
        diffs.append(tau_mean(A, draw) - tau_mean(B, draw))
    diffs.sort()
    return point, diffs[int(0.025 * B_iter)], diffs[int(0.975 * B_iter) - 1], len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--a", required=True, help="method/config, e.g. farsel/W=1024_keep=prompt_n=16_score=mid3")
    ap.add_argument("--b", required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--cuts", default=os.path.join(ROOT, "results/loop_cuts.csv"))
    ap.add_argument("--iters", type=int, default=10000)
    args = ap.parse_args()
    cuts = load_cuts(args.cuts) if os.path.exists(args.cuts) else {}
    runs = [r for r in load_raw(args.raw) if not args.ids or r["header"]["trace_id"] in args.ids]
    A = per_trace(runs, tuple(args.a.split("/", 1)), cuts)
    B = per_trace(runs, tuple(args.b.split("/", 1)), cuts)
    d, lo, hi, n = paired_bootstrap(A, B, args.iters)
    print(f"Δτ = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n = {n} paired traces, B = {args.iters})")


if __name__ == "__main__":
    main()
