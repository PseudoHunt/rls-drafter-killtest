"""Per-slot conditional acceptance P(a >= i | a >= i-1) from dev replay logs (truncated variant).

A round counts toward slot i only if the (possibly truncated) trace had a token for slot i, i.e. i <= min(15,
end - pos) with end = cut - 1 for flagged traces (n_out - 1 otherwise). Groups pool counts over their slots:
P_group = sum_i #(a >= i) / sum_i #(a >= i-1). Writes results/slot_accept.md and prints it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import DATASETS, load_cuts, load_raw, select  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = [("baseline", "default", "baseline"),
        ("rls", "dtype=fp64_lam=0.9999_n0=128", "best RLS (n0=128, λ=0.9999)"),
        ("bias", "rho=0.999", "best hidden-residual bias (ρ=0.999)")]
GROUPS = (("slot 1", 1, 1), ("slots 2–4", 2, 4), ("slots 5–15", 5, 15))
S = 15


def counts(runs, cuts):
    """{dataset: (num[i], den[i]) for i = 1..S}"""
    out = {}
    for run in select(runs, "trunc", cuts):
        h = run["header"]
        c = cuts.get(h["trace_id"]) if h.get("loop") else None
        end = (c - 1) if c is not None else h["n_out"] - 1
        num, den = out.setdefault(h["dataset"], ([0] * (S + 1), [0] * (S + 1)))
        for r in run["rounds"]:
            avail = min(S, end - r["pos"])
            for i in range(1, avail + 1):
                if r["a"] >= i - 1:
                    den[i] += 1
                    num[i] += r["a"] >= i
    return out


def group_p(nd, lo, hi):
    num, den = nd
    d = sum(den[lo:hi + 1])
    return sum(num[lo:hi + 1]) / d if d else float("nan")


def main():
    cuts = load_cuts(os.path.join(ROOT, "results/loop_cuts.csv"))
    runs = load_raw(os.path.join(ROOT, "results/raw"), "dev")
    L = ["## Per-slot conditional acceptance P(a ≥ i | a ≥ i−1) — dev, flagged traces truncated", "",
         "Counts pooled over rounds within a dataset; mean = average of AIME and LCB. A round counts toward slot i "
         "only if the (truncated) trace had a token at slot i.", "",
         "| arm | " + " | ".join(f"{g} {ds}" for g, _, _ in GROUPS for ds in ("AIME", "LCB", "mean")) + " |",
         "|---|" + "---|" * (3 * len(GROUPS))]
    for method, cfg, label in ARMS:
        sel = [r for r in runs if r["header"]["method"] == method and r["header"]["config"] == cfg]
        cnt = counts(sel, cuts)
        cells = []
        for _, lo, hi in GROUPS:
            ps = [group_p(cnt[ds], lo, hi) for ds in DATASETS]
            cells += [f"{p:.3f}" for p in ps] + [f"{sum(ps) / len(ps):.3f}"]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    open(os.path.join(ROOT, "results/slot_accept.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
