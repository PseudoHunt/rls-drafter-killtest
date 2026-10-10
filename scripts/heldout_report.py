"""Held-out test report (pre-registered in RUNLOG.md, 2026-10-10) -> results/heldout.md, held-out section of summary.md.

  --gate-only      print "PASS d=…" / "FAIL d=…" for τ_window − τ_base ≥ 0.4 (test, decision variant)
  --estimate-t2b   print the projected hours for S2 + TTS stride 1 on the T2 traces (from the measured hybrid cost)

Statistics: paired bootstrap over traces (stratified by dataset, same draw for all arms), B = 10,000, seed 0, 95%
percentile intervals; τ = mean over AIME and LCB of the pooled τ, decision variant (flagged traces truncated at loop
onset, dropped traces excluded). R = ratio of mean gains, (τ_arm − τ_base)/(τ_TTS − τ_base), on the same draws.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import DATASETS, load_cuts, load_raw, select, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = "1e-05"
A = {
    "baseline": ("baseline", "default"),
    "window": ("window", "W=1024_keep=prompt"),
    "S2": ("farsel", "W=1024_keep=prompt_n=16_score=mid3"),
    "C1": ("window", "W=1536_keep=prompt"),
    "C2": ("farsel", "W=1024_keep=prompt_n=16_score=random_seed=0"),
    "TTS": ("tts", f"lr={LR}"),
    "TTS stride 10": ("tts", f"lr={LR}_stride=10"),
    "hybrid": ("tts", f"ctx=s2_lr={LR}_pool=True_stride=10"),
    "S2+TTS": ("tts", f"ctx=s2_lr={LR}"),
}
CTX = {"TTS": "full", "TTS stride 10": "full", "hybrid": "S2", "S2+TTS": "S2"}
A30_BW = 933e9  # A30 HBM2 bytes/s (spec)


def per_trace(runs, key, cuts, variant="trunc"):
    out = {}
    for run in select([r for r in runs if (r["header"]["method"], r["header"]["config"]) == key], variant, cuts):
        rs = run["rounds"]
        out[run["header"]["trace_id"]] = (run["header"]["dataset"], sum(r["tau"] for r in rs), len(rs))
    return out


def tau_of(stats, ids):
    by = {ds: [0, 0] for ds in DATASETS}
    for i in ids:
        ds, tok, n = stats[i]
        by[ds][0] += tok
        by[ds][1] += n
    return sum(by[ds][0] / by[ds][1] for ds in DATASETS if by[ds][1]) / sum(1 for ds in DATASETS if by[ds][1])


class Boot:
    """Shared stratified resamples of a trace set."""

    def __init__(self, ids, ds_of, iters, seed=0):
        rng = random.Random(seed)
        strata = {ds: sorted(i for i in ids if ds_of[i] == ds) for ds in DATASETS}
        self.ids = sorted(ids)
        self.draws = [[rng.choice(strata[ds]) for ds in DATASETS for _ in strata[ds]] for _ in range(iters)]

    def ci(self, f):
        point = f(self.ids)
        vals = sorted(f(d) for d in self.draws)
        n = len(vals)
        return point, vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def fmt_ci(p, lo, hi, n=3):
    return f"{p:+.{n}f} [{lo:+.{n}f}, {hi:+.{n}f}]"


def round_rows(runs, key, ids=None):
    return [r for run in runs if (run["header"]["method"], run["header"]["config"]) == key
            and (ids is None or run["header"]["trace_id"] in ids) for r in run["rounds"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results/raw_test"))
    ap.add_argument("--iso", default=os.path.join(ROOT, "results/timing_isolated_a30"))
    ap.add_argument("--cuts", default=os.path.join(ROOT, "results/loop_cuts.csv"))
    ap.add_argument("--t2-ids", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/heldout.md"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "results/summary.md"))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--stage", default="done")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--estimate-t2b", action="store_true")
    args = ap.parse_args()
    cuts = load_cuts(args.cuts) if os.path.exists(args.cuts) else {}
    runs = load_raw(args.raw, "test")
    have = {k for k, key in A.items() if any((r["header"]["method"], r["header"]["config"]) == key for r in runs)}
    T = {k: per_trace(runs, A[k], cuts) for k in have}

    if args.gate_only:
        ids = sorted(set(T["baseline"]) & set(T["window"]))
        d = tau_of(T["window"], ids) - tau_of(T["baseline"], ids)
        print(("PASS" if d >= 0.4 else "FAIL") + f" d={d:.4f} n={len(ids)}")
        return
    if args.estimate_t2b:
        hy = round_rows(runs, A["hybrid"], set(args.t2_ids))
        ms = sum(r["draft_ms"] + r["update_ms"] for r in hy) / max(len(hy), 1)
        tts1 = round_rows(runs, A["TTS"], set(args.t2_ids))
        step = max(sum(r["update_ms"] for r in tts1) / max(len(tts1), 1) - sum(r["update_ms"] for r in hy) / max(len(hy), 1), 0)
        hours = len(hy) * (ms + step) * 1.15 / 3.6e6 + len(args.t2_ids) * 30 / 3600
        print(f"{hours:.3f}")
        return

    ds_of = {i: v[0] for k in T for i, v in T[k].items()}
    L = [f"# Held-out test (A30) — stage: {args.stage}", "",
         "Pre-registered in RUNLOG.md (2026-10-10). Decision variant: flagged traces truncated at loop onset, dropped "
         "traces excluded; τ = mean over AIME and LCB of the pooled τ. CIs: paired bootstrap over traces (stratified by "
         f"dataset), B = {args.iters}, seed 0, 95% percentile. TTS setting (T0): lr = 1e-5, stale cache (best stale lr "
         "on the AIME dev grid: 1e-5 4.487, 3e-5 4.425, 1e-4 3.596 over 5 traces; refresh check not run).", ""]

    # ---- per-arm tables (all test traces available per arm; T2 arms on the 16)
    tabs = {v: table(runs, v, cuts) for v in ("trunc", "exclude", "loops")}
    L += ["## τ per arm", "", "| arm | traces | τ | τ AIME | τ LCB | τ <2K | τ ≥8K | τ excl. flagged | τ loops incl. |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, key in A.items():
        if k not in have:
            L.append(f"| {k} | not run | | | | | | | |")
            continue
        t, e, lp = tabs["trunc"][key], tabs["exclude"][key], tabs["loops"][key]
        g = lambda tb, f: tb["mean"][f] if "mean" in tb else float("nan")  # noqa: E731
        L.append(f"| {k} | {len(T[k])} | {g(t, 'tau'):.3f} | {t['aime']['tau']:.3f} | {t['lcb']['tau']:.3f} | "
                 f"{g(t, 'tau_lt2k'):.3f} | {g(t, 'tau_ge8k'):.3f} | {g(e, 'tau'):.3f} | {g(lp, 'tau'):.3f} |")
    L.append("")

    # ---- contrasts
    claims = []
    res = {}
    def contrast(name, a, b, ids):  # noqa: E306
        if a not in have or b not in have:
            return None
        ids = sorted(set(ids) & set(T[a]) & set(T[b]))
        bt = Boot(ids, ds_of, args.iters)
        r = bt.ci(lambda d: tau_of(T[a], d) - tau_of(T[b], d))
        res[name] = r + (len(ids),)
        return r

    all50 = set(T.get("baseline", {}))
    L += ["## Contrasts (paired bootstrap 95% CI)", "", "| contrast | traces | Δτ [95% CI] |", "|---|---|---|"]
    for name, a, b, ids in [("window − base", "window", "baseline", all50), ("S2 − base", "S2", "baseline", all50),
                            ("TTS − base", "TTS", "baseline", all50), ("S2 − window", "S2", "window", all50),
                            ("S2 − C1", "S2", "C1", all50), ("S2 − C2", "S2", "C2", all50),
                            ("S2 − TTS(stride 1)", "S2", "TTS", all50),
                            ("hybrid − TTS(stride 1)", "hybrid", "TTS", args.t2_ids),
                            ("(S2 + TTS stride 1) − S2", "S2+TTS", "S2", args.t2_ids),
                            ("TTS stride 10 − TTS stride 1", "TTS stride 10", "TTS", args.t2_ids)]:
        r = contrast(name, a, b, ids)
        L.append(f"| {name} | {res[name][3] if r else '–'} | {fmt_ci(*r) if r else 'not available'} |")
    Rs = {}
    if {"baseline", "TTS"} <= have:
        for arm in ("window", "S2"):
            if arm in have:
                ids = sorted(all50 & set(T[arm]) & set(T["TTS"]))
                bt = Boot(ids, ds_of, args.iters)
                f = lambda d, arm=arm: ((tau_of(T[arm], d) - tau_of(T["baseline"], d))  # noqa: E731
                                        / (tau_of(T["TTS"], d) - tau_of(T["baseline"], d)))
                Rs[arm] = bt.ci(f)
                L.append(f"| R_{arm} = gain_{arm} / gain_TTS | {len(ids)} | {fmt_ci(*Rs[arm])} |")
    L.append("")

    # ---- claims
    def ok(name, cond):
        claims.append((name, cond))
    if "window" in Rs:
        ok("Headline: window R ≥ 0.5", f"R = {Rs['window'][0]:.3f} → {'**YES**' if Rs['window'][0] >= 0.5 else '**NO**'}")
    else:
        ok("Headline: window R ≥ 0.5", "not evaluable (TTS not run)")
    if all(k in res for k in ("S2 − window", "S2 − C1", "S2 − C2")):
        sw, s1, s2 = res["S2 − window"], res["S2 − C1"], res["S2 − C2"]
        adds = sw[0] >= 0.05 and sw[1] > 0 and s1[1] > 0 and s2[1] > 0
        ok("S2 adds value: S2 − window ≥ 0.05 with CI > 0, and CIs of S2 − C1, S2 − C2 exclude 0 (above)",
           f"{fmt_ci(*sw[:3])}; C1 {fmt_ci(*s1[:3])}; C2 {fmt_ci(*s2[:3])} → {'**YES**' if adds else '**NO**'}")
    else:
        adds = False
        ok("S2 adds value", "not evaluable (an arm is missing)")
    if "S2 − TTS(stride 1)" in res:
        d = res["S2 − TTS(stride 1)"]
        ok("S2 dominates TTS: lower CI bound of S2 − TTS(stride 1) ≥ 0",
           f"{fmt_ci(*d[:3])} → {'**YES**' if d[1] >= 0 else '**NO**'}")
    else:
        ok("S2 dominates TTS", "not evaluable")
    upd = {}
    for k in ("TTS", "TTS stride 10", "hybrid", "S2+TTS"):
        if k in have:
            rr = round_rows(runs, A[k], set(args.t2_ids))
            g_ = [r["update_ms"] for r in rr if r.get("grad", 1)]
            upd[k] = (sum(r["update_ms"] for r in rr) / max(len(rr), 1), sum(g_) / max(len(g_), 1),
                      sum(r["draft_ms"] for r in rr) / max(len(rr), 1), len(rr))
    if "hybrid − TTS(stride 1)" in res and "hybrid" in upd and "TTS" in upd:
        d = res["hybrid − TTS(stride 1)"]
        ratio = upd["hybrid"][0] / upd["TTS"][0]
        yes = d[1] > 0 and ratio <= 0.10
        ok("Hybrid: hybrid − TTS(stride 1) CI above 0, at ≤ 10% of TTS(stride 1)'s update ms/round (16 traces)",
           f"{fmt_ci(*d[:3])}; update ms ratio = {upd['hybrid'][0]:.1f}/{upd['TTS'][0]:.1f} = {ratio:.1%} → "
           f"{'**YES**' if yes else '**NO**'}")
    else:
        ok("Hybrid", "not evaluable")
    if "(S2 + TTS stride 1) − S2" in res:
        ok("Residual after S2 (report only)", fmt_ci(*res["(S2 + TTS stride 1) − S2"][:3]))
    L += ["## Pre-registered claims", ""] + [f"- **{n}**: {c}" for n, c in claims] + [""]

    # ---- costs
    L += ["## Costs", "", "TTS variants on the 16 T2 traces (one TTS process alone on the GPU):", "",
          "| variant | context | update ms/round | update ms per gradient round | draft ms/round | rounds |",
          "|---|---|---|---|---|---|"]
    for k, (u, ug, dm_, n) in upd.items():
        L.append(f"| {k} | {CTX[k]} | {u:.2f} | {ug:.2f} | {dm_:.2f} | {n} |")
    if "S2" in have:
        rr = [(run["header"]["prompt_len"] + r["pos"]) for run in runs
              if (run["header"]["method"], run["header"]["config"]) == A["S2"] for r in run["rounds"]]
        keys = sum(rr) / max(len(rr), 1)
        b = 3 * 8 * 128 * 2 * keys
        L += ["", f"S2 extra target-side cost (one query row against the K cache of target layers 17/25/33, bytes read): "
              f"3 layers × 8 KV heads × 128 × 2 B × t'−1 keys = {b / 1e6:.1f} MB per round on average over test rounds "
              f"(mean t'−1 = {keys:.0f}) → ≈ {b / A30_BW * 1e3:.3f} ms per round at the A30's 933 GB/s; at 16K context "
              f"{3 * 8 * 128 * 2 * 16384 / 1e6:.1f} MB ≈ {3 * 8 * 128 * 2 * 16384 / A30_BW * 1e3:.3f} ms "
              "(microbenchmark, unfused: results/attn_row_cost.md)."]
    if os.path.isdir(args.iso):
        iso = load_raw(args.iso)
        L += ["", "Drafting ms/round, isolated (one process alone on the GPU, 2 dev traces):", "",
              "| arm | draft ms/round | rounds |", "|---|---|---|"]
        for k in ("baseline", "window", "S2"):
            rr = round_rows(iso, A[k])
            if rr:
                L.append(f"| {k} | {sum(r['draft_ms'] for r in rr) / len(rr):.2f} | {len(rr)} |")
    L.append("")
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))

    # ---- summary.md: held-out section at the top (replaced on re-runs)
    if os.path.exists(args.summary):
        s = open(args.summary).read()
        head, _, rest = s.partition("\n")
        beg, end = "<!-- HELDOUT BEGIN -->", "<!-- HELDOUT END -->"
        if beg in rest:
            rest = rest.split(beg)[0] + rest.split(end, 1)[1]
        lead = "S2 and the window" if adds else "the window"
        sec = [beg, "", f"## Held-out test (A30, {len(all50)} test traces) — leads with {lead}", ""]
        if "window" in Rs:
            sec.append(f"- **Window (W 1024 + prompt): R = {Rs['window'][0]:.3f}** "
                       f"[{Rs['window'][1]:.3f}, {Rs['window'][2]:.3f}]; window − base "
                       f"{fmt_ci(*res['window − base'][:3]) if 'window − base' in res else '–'}.")
        if "S2" in Rs:
            sec.append(f"- S2: R = {Rs['S2'][0]:.3f} [{Rs['S2'][1]:.3f}, {Rs['S2'][2]:.3f}].")
        sec += [f"- {n}: {c}" for n, c in claims] + ["", f"Full report: `results/heldout.md` (stage: {args.stage}).", "", end]
        open(args.summary, "w").write(head + "\n\n" + "\n".join(sec) + "\n" + rest.lstrip("\n"))


if __name__ == "__main__":
    main()
