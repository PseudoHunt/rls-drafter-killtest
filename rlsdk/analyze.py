"""Aggregate per-round logs: pooled tau per dataset (non-looping traces), position-sliced tau, R, timings."""
import glob
import json
import os
from collections import defaultdict

BIN = 2048
DATASETS = ("aime", "lcb")


def load_raw(raw_dir, split=None):
    """Return list of runs: {header, rounds, summary}."""
    runs = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        lines = [json.loads(line) for line in open(p)]
        head, summ = lines[0], lines[-1]
        if split and head["split"] != split:
            continue
        runs.append({"header": head, "rounds": [r for r in lines if r["type"] == "round"], "summary": summ})
    return runs


def sigma2_from_baseline(runs):
    s = sum(r["summary"]["z2_sum"] for r in runs if r["header"]["method"] == "baseline")
    n = sum(r["summary"]["z2_count"] for r in runs if r["header"]["method"] == "baseline")
    return s / n


def _pooled(rounds, lo=None, hi=None):
    sel = [r for r in rounds if (lo is None or r["pos"] >= lo) and (hi is None or r["pos"] < hi)]
    return (sum(r["tau"] for r in sel) / len(sel)) if sel else float("nan"), len(sel)


def apply_cut(rounds, c):
    """Rounds of a replay over the output truncated to its first c tokens, derived from the full-trace replay.

    Exact: rounds only depend on the past, the truncated trace ends at output index c-1, so its rounds are the
    full replay's rounds with pos < c-1 and tau clipped to (c-1-pos) (DFlash's end-of-trace clipping).
    Verified against a real truncated replay in tests/test_replay.py::test_truncation_derivation_exact.
    """
    out = []
    for r in rounds:
        if r["pos"] >= c - 1:
            break
        out.append({**r, "tau": min(r["tau"], c - 1 - r["pos"])})
    return out


def load_cuts(path):
    """loop_cuts.csv -> {trace_id: c or None (dropped)}."""
    import csv
    return {row["trace_id"]: (None if row["dropped"] == "True" else int(row["c"])) for row in csv.DictReader(open(path))}


def select(runs, variant, cuts=None):
    """variant: 'trunc' (decision metric, Amendment 2), 'exclude' (flagged dropped), 'loops' (all, untruncated)."""
    out = []
    for run in runs:
        h = run["header"]
        if not h.get("loop") or variant == "loops":
            out.append(run)
        elif variant == "trunc":
            c = cuts[h["trace_id"]]
            if c is not None:
                out.append({**run, "rounds": apply_cut(run["rounds"], c)})
    return out


def table(runs, variant="trunc", cuts=None):
    """{(method, config): {dataset: metrics, 'mean': metrics}}"""
    groups = defaultdict(lambda: defaultdict(list))
    for run in select(runs, variant, cuts):
        h = run["header"]
        groups[(h["method"], h["config"])][h["dataset"]].append(run)
    out = {}
    for key, by_ds in groups.items():
        res = {}
        for ds, rs in by_ds.items():
            rounds = [r for run in rs for r in run["rounds"]]
            n = max(len(rounds), 1)
            res[ds] = {
                "tau": _pooled(rounds)[0],
                "tau_ge8k": _pooled(rounds, lo=8192)[0],
                "tau_lt2k": _pooled(rounds, hi=2048)[0],
                "n_rounds": len(rounds), "n_traces": len(rs),
                "draft_ms": sum(r["draft_ms"] for r in rounds) / n,
                "update_ms": sum(r["update_ms"] for r in rounds) / n,
                "solve_ms": sum(r.get("solve_ms", 0.0) for r in rounds) / n,
                "state_mb": max(run["header"]["state_bytes"] for run in rs) / 2**20,
                "bins": {b: _pooled(rounds, b * BIN, (b + 1) * BIN) for b in range(8)},
            }
        if all(ds in res for ds in DATASETS):
            res["mean"] = {k: sum(res[ds][k] for ds in DATASETS) / len(DATASETS)
                           for k in ("tau", "tau_ge8k", "tau_lt2k", "draft_ms", "update_ms", "solve_ms", "state_mb")}
        out[key] = res
    return out


def gain_recovered(tab, rls_key, tts_key, base_key, metric="tau", ds="mean"):
    b, r, t = tab[base_key][ds][metric], tab[rls_key][ds][metric], tab[tts_key][ds][metric]
    return (r - b) / (t - b) if t != b else float("nan")


def best_per_method(tab, ds="mean"):
    best = {}
    for (method, cfg), res in tab.items():
        if ds not in res:
            continue
        if method not in best or res[ds]["tau"] > tab[(method, best[method])][ds]["tau"]:
            best[method] = cfg
    return best


def plot_bins(tab, keys, labels, ds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in zip(keys, labels):
        bins = tab[key][ds]["bins"]
        xs = [b * 2 + 1 for b in sorted(bins) if bins[b][1] > 0]
        ys = [bins[b][0] for b in sorted(bins) if bins[b][1] > 0]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("output position (K tokens, 2K bins)")
    ax.set_ylabel("τ (committed tokens / round)")
    ax.set_title(f"τ vs output position — {ds.upper()}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
