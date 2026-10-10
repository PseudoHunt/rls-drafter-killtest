"""VPA report (addendum B, RUNLOG.md) -> results/vpa.md and a VPA section in summary.md after the held-out section.

  --pick               best screen config on the 4 reference traces by mean τ (prints its config name)
  --promote CONFIG     promotion rule on the 7 confirmation dev traces: VPA − S2 ≥ +0.05 τ (mean) AND VPA ≥ S2 on
                       ≥ 75% of traces -> prints "PROMOTE …" or "NO …"
  (default)            write the report (screen, confirmation, T3 contrasts if run)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heldout_report import Boot, fmt_ci, per_trace, round_rows, tau_of  # noqa: E402
from rlsdk.analyze import load_cuts, load_raw, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
REF = ["aime24_II_14", "aime24_I_10", "lcb_abc393_d", "lcb_arc191_d"]
CONF = ["aime24_II_02", "aime24_II_09", "aime24_II_13", "aime24_I_13", "lcb_3701", "lcb_arc193_d", "lcb_arc195_c"]
GRID = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)]
VPA = [("vpa", f"M=64_beta={b}_lam={l}") for b, l in GRID]
BASE, WIN = ("baseline", "default"), ("window", "W=1024_keep=prompt")
S2 = ("farsel", "W=1024_keep=prompt_n=16_score=mid3")
TTS = ("tts", "lr=1e-05")


def tab_for(raw, ids, cuts):
    runs = [r for r in load_raw(raw) if r["header"]["trace_id"] in ids]
    return runs, table(runs, "trunc", cuts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--promote")
    ap.add_argument("--stage", default="done")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--root", default=None, help="dry runs: alternative tree with results/ (and summary.md)")
    args = ap.parse_args()
    if args.root:
        global J
        J = lambda *p: os.path.join(args.root, *p)  # noqa: E731
    cuts = load_cuts(J("results/loop_cuts.csv"))
    sruns, stab = tab_for(J("results/raw_a30_window"), REF, cuts)
    have = [k for k in VPA if k in stab and "mean" in stab[k]]
    if args.pick:
        best = max(have, key=lambda k: (stab[k]["mean"]["tau"], -VPA.index(k)))
        print(best[1])
        return
    cruns, ctab = tab_for(J("results/raw_a30_dev"), CONF, cuts) if os.path.isdir(J("results/raw_a30_dev")) else ([], {})
    if args.promote:
        V = ("vpa", args.promote)
        d = ctab[V]["mean"]["tau"] - ctab[S2]["mean"]["tau"]
        pv, ps = per_trace(cruns, V, cuts), per_trace(cruns, S2, cuts)
        ids = sorted(set(pv) & set(ps))
        wins = sum(pv[i][1] / pv[i][2] >= ps[i][1] / ps[i][2] for i in ids)
        ok = d >= 0.05 and wins >= 0.75 * len(ids)
        print(("PROMOTE" if ok else "NO") + f" d={d:+.4f} wins={wins}/{len(ids)}")
        return

    def row(tb, k, label):
        r, m = tb[k], tb[k]["mean"]
        return (f"| {label} | {m['tau']:.3f} | {r['aime']['tau']:.3f} | {r['lcb']['tau']:.3f} | {m['tau_ge8k']:.3f} | "
                f"{m['draft_ms']:.2f} |")

    L = [f"# Verifier-Prior Attention (VPA) — stage: {args.stage}", "",
         "Pre-registered: RUNLOG.md addendum B. Kept context = prompt + last W0 = 1024 + top M far 32-token chunks (S2's "
         "signal); bias on kept context keys b_j = β·log(m_j + 1/N_ctx) − λ·log(1 + max(0, d_j − W0)/W0) (distance term on "
         "far keys only), centred; block keys unbiased. τ: decision variant, mean of AIME and LCB pooled τ. Draft ms under "
         "2 concurrent workers (indicative).", "",
         "## Step 1 — screen (4 reference traces), M = 64", "",
         "| config | τ | τ AIME | τ LCB | τ ≥8K | draft ms/round |", "|---|---|---|---|---|---|"]
    for k, lab in [(BASE, "baseline"), (WIN, "window (W 1024 + prompt)"), (S2, "S2 (M = 16, no bias)")]:
        if k in stab:
            L.append(row(stab, k, lab))
    for k in VPA:
        if k in stab and "mean" in stab[k]:
            L.append(row(stab, k, f"VPA {k[1]}"))
    best = max(have, key=lambda k: (stab[k]["mean"]["tau"], -VPA.index(k))) if have else None
    L += ["", f"Chosen (best mean τ): **{best[1] if best else '–'}**.", ""]
    promo_line = "not reached"
    if ctab and best and best in ctab:
        L += ["## Step 2 — confirmation (7 non-dropped dev traces outside the screen set)", "",
              "| config | τ | τ AIME | τ LCB | τ ≥8K | draft ms/round |", "|---|---|---|---|---|---|"]
        for k, lab in [(WIN, "window"), (S2, "S2"), (best, f"VPA {best[1]}")]:
            if k in ctab:
                L.append(row(ctab, k, lab))
        pv, ps, pw = (per_trace(cruns, k, cuts) for k in (best, S2, WIN))
        L += ["", "| trace | window τ | S2 τ | VPA τ | VPA ≥ S2 |", "|---|---|---|---|---|"]
        wins = 0
        for i in CONF:
            if i in pv and i in ps:
                tv, tsv = pv[i][1] / pv[i][2], ps[i][1] / ps[i][2]
                wins += tv >= tsv
                L.append(f"| {i} | {pw[i][1] / pw[i][2]:.3f} | {tsv:.3f} | {tv:.3f} | {'yes' if tv >= tsv else 'no'} |")
        d = ctab[best]["mean"]["tau"] - ctab[S2]["mean"]["tau"]
        ok = d >= 0.05 and wins >= 0.75 * len(pv)
        promo_line = f"VPA − S2 = {d:+.3f} (need ≥ +0.05), VPA ≥ S2 on {wins}/{len(pv)} (need ≥ 75%) → " + \
            ("**PROMOTED**" if ok else "**not promoted**")
        L += ["", f"Promotion rule: {promo_line}", ""]
    claims = []
    truns = load_raw(J("results/raw_test"), "test") if os.path.isdir(J("results/raw_test")) else []
    if best and any((r["header"]["method"], r["header"]["config"]) == best for r in truns):
        T = {k: per_trace(truns, key, cuts) for k, key in (("VPA", best), ("S2", S2), ("window", WIN), ("TTS", TTS))}
        ttab = table(truns, "trunc", cuts)
        ds_of = {i: v[0] for k in T for i, v in T[k].items()}
        L += ["## T3 — VPA on the test traces", "", "| arm | τ | τ AIME | τ LCB | τ ≥8K | draft ms/round |",
              "|---|---|---|---|---|---|"]
        for k, lab in [(best, f"VPA {best[1]}"), (S2, "S2"), (WIN, "window"), (TTS, "TTS stride 1")]:
            if k in ttab:
                L.append(row(ttab, k, lab))
        L += ["", "| contrast | traces | Δτ [95% CI] |", "|---|---|---|"]
        res = {}
        for name, b in (("VPA − S2", "S2"), ("VPA − window", "window"), ("VPA − TTS(stride 1)", "TTS")):
            if not T[b]:
                continue
            ids = sorted(set(T["VPA"]) & set(T[b]))
            res[name] = Boot(ids, ds_of, args.iters).ci(lambda dr, b=b: tau_of(T["VPA"], dr) - tau_of(T[b], dr))
            L.append(f"| {name} | {len(ids)} | {fmt_ci(*res[name])} |")
        if "VPA − S2" in res:
            r = res["VPA − S2"]
            claims.append(f"VPA improves on S2 (CI excludes 0, above it): {fmt_ci(*r)} → "
                          + ("**YES**" if r[1] > 0 else ("**NO** (CI below 0: VPA is worse)" if r[2] < 0 else "**NO**")))
        if "VPA − TTS(stride 1)" in res:
            r = res["VPA − TTS(stride 1)"]
            claims.append(f"VPA dominates TTS (lower CI ≥ 0): {fmt_ci(*r)} → {'**YES**' if r[1] >= 0 else '**NO**'}")
        L += ["", "Claims:"] + [f"- {c}" for c in claims] + [""]
    elif "not promoted" in promo_line:
        L += ["Not promoted: dev results reported as is. The window (M = 0, β = λ = 0) and S2 (M = 16, β = λ = 0) are "
              "limits of the VPA family; on dev the hard-selection limit performed best.", ""]
    open(J("results/vpa.md"), "w").write("\n".join(L))
    print("\n".join(L))

    s = open(J("results/summary.md")).read()
    beg, end = "<!-- VPA BEGIN -->", "<!-- VPA END -->"
    if beg in s:
        s = s.split(beg)[0].rstrip("\n") + "\n\n" + s.split(end, 1)[1].lstrip("\n")
    sec = "\n".join([beg, "", f"## Verifier-Prior Attention (VPA; dev-tuned, stage: {args.stage})", "",
                     f"- Screen (4 reference traces): best = {best[1] if best else '–'}"
                     + (f", τ = {stab[best]['mean']['tau']:.3f} (S2 {stab[S2]['mean']['tau']:.3f}, window "
                        f"{stab[WIN]['mean']['tau']:.3f})" if best and S2 in stab and WIN in stab else "") + ".",
                     f"- Confirmation (7 dev traces): {promo_line}."] + [f"- {c}" for c in claims]
                    + ["", "Full report: `results/vpa.md`.", "", end, ""])
    marker = "<!-- HELDOUT END -->"
    if marker in s:
        a, b = s.split(marker, 1)
        s = a + marker + "\n\n" + sec + b.lstrip("\n")
    else:
        head, _, rest = s.partition("\n")
        s = head + "\n\n" + sec + rest.lstrip("\n")
    open(J("results/summary.md"), "w").write(s)


if __name__ == "__main__":
    main()
