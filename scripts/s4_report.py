"""Session-4 reports (pre-registered in RUNLOG.md, 2026-10-11 08:55 UTC): SharpDraft anchor (item 1d), held-out arms
(items 1e, 2, 3) with τ overall / per bin / per dataset and paired CIs vs TTS stride 1 and S2+cal, scale diagnostics.

  python scripts/s4_report.py anchor                 -> results/s4_anchor.md (+ PASS/FAIL line)
  python scripts/s4_report.py item1                  -> results/s4_sharpdraft.md
  python scripts/s4_report.py item2                  -> results/s4_thinking.md
  python scripts/s4_report.py item3                  -> results/s4_calfull.md

Same procedure as scripts/s3_report.py (decision variant, Boot: paired, stratified by dataset, B = 10,000, seed 0).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_report import BINS, REF, S2, TTS1, B10, Arm, R, cuts, tau, Boot, fmt_ci  # noqa: E402
from rlsdk.analyze import DATASETS  # noqa: E402

BASE = ("baseline", "default")
SD = ("sharp", "N0=2048_ctx=full_k=32_logcap=0.5_s=5")
SD_S2 = ("sharp", "N0=2048_ctx=s2_k=32_logcap=0.5_s=5")
CAP = math.exp(0.5)


class Composite:
    """An arm whose rounds come from different replays per dataset (SD+S2+cal: LCB from the new run, AIME from the
    S2+cal replay, identical by construction)."""

    def __init__(self, label, by_ds):
        self.label, self.by_ds = label, by_ds
        self.dec, self.ds = {}, {}
        for ds, arm in by_ds.items():
            for t, r in arm.dec.items():
                if arm.ds[t] == ds:
                    self.dec[t], self.ds[t] = r, ds
        self.full = [r for a in by_ds.values() for r in a.full if r["header"]["dataset"] in by_ds
                     and by_ds[r["header"]["dataset"]] is a]

    stats = Arm.stats
    update_ms = Arm.update_ms
    draft_ms = Arm.draft_ms


def test_ids():
    return sorted(json.loads(x)["trace_id"] for x in open(R("traces/traces.jsonl")) if json.loads(x)["split"] == "test")


def bin_cells(a, ids):
    return [f"{tau(a.stats(), ids):.3f}"] + [f"{tau(a.stats(lo, hi), ids):.3f}" for _, lo, hi in BINS]


def heldout_tables(arms, refs, ids, title, notes=()):
    boot = Boot(ids, refs[0].ds, 10000)
    by_ds = {ds: sorted(i for i in ids if refs[0].ds[i] == ds) for ds in DATASETS}
    boots = {ds: Boot(by_ds[ds], refs[0].ds, 10000) for ds in DATASETS}
    L = [f"# {title}", ""] + list(notes) + ["",
         "| arm | τ | " + " | ".join(f"τ {b[0]}" for b in BINS) + " | update ms/round | draft ms/round* | rounds |",
         "|---|---|" + "---|" * len(BINS) + "---|---|---|"]
    for a in refs + arms:
        u, n = a.update_ms()
        u = 0.0 if a.label in ("S2+cal", "S2", "baseline") else u
        L.append(f"| {a.label} | " + " | ".join(bin_cells(a, ids)) + f" | {u:.2f} | {a.draft_ms():.1f} | {n} |")
    L += ["", "\\* draft ms under 2 concurrent light workers for the new arms: indicative only.", "",
          "## Contrasts (paired 95% CI)", "",
          "| contrast | overall | " + " | ".join(b[0] for b in BINS) + " | AIME | LCB |",
          "|---|---|" + "---|" * len(BINS) + "---|---|"]
    for a in arms:
        for b in refs:
            sa, sb = a.stats(), b.stats()
            cells = [fmt_ci(*boot.ci(lambda d: tau(sa, d) - tau(sb, d)))]
            for _, lo, hi in BINS:
                xa, xb = a.stats(lo, hi), b.stats(lo, hi)
                cells.append(fmt_ci(*boot.ci(lambda d: tau(xa, d) - tau(xb, d))))
            cells += [fmt_ci(*boots[ds].ci(lambda d: tau(sa, d) - tau(sb, d))) for ds in DATASETS]
            L.append(f"| {a.label} − {b.label} | " + " | ".join(cells) + " |")
    L += ["", "## Per dataset (95% CI, bootstrap within the dataset)", "", "| arm | τ AIME | τ LCB |", "|---|---|---|"]
    for a in refs + arms:
        st = a.stats()
        cells = []
        for ds in DATASETS:
            p_, lo, hi = boots[ds].ci(lambda d: tau(st, d))
            cells.append(f"{p_:.3f} [{lo:.3f}, {hi:.3f}] (n={len(by_ds[ds])})")
        L.append(f"| {a.label} | " + " | ".join(cells) + " |")
    return L, boot


def reading(boot, a, b):
    sa, sb = a.stats(), b.stats()
    p_, lo, hi = boot.ci(lambda d: tau(sa, d) - tau(sb, d))
    word = "beats" if lo > 0 else ("below" if hi < 0 else "within the CI of")
    return word, (p_, lo, hi)


def scale_diag(arm, ids, label):
    """β̂ and N distribution over the decision-variant rounds, per dataset."""
    L = [f"### Scale diagnostics: {label}", "",
         "| dataset | rounds | β̂ mean | β̂ max | at β̂ = 1 | at cap e^0.5 | N median | N p90 | N max |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ds in list(DATASETS) + ["all"]:
        rs = [x for t in ids if t in arm.dec and (ds == "all" or arm.ds[t] == ds) for x in arm.dec[t]["rounds"]
              if "qscale" in x]
        if not rs:
            continue
        b = [x["qscale"] for x in rs]
        n = sorted(x["n_vis"] for x in rs)
        L.append(f"| {ds} | {len(rs)} | {sum(b) / len(b):.3f} | {max(b):.3f} | "
                 f"{100 * sum(v == 1.0 for v in b) / len(b):.1f}% | {100 * sum(v >= CAP for v in b) / len(b):.1f}% | "
                 f"{n[len(n) // 2]} | {n[int(0.9 * len(n))]} | {n[-1]} |")
    return L


def anchor(args):
    c = cuts()
    base = Arm("baseline", R("results/raw_a30"), BASE, REF, c)
    sd = Arm("SharpDraft", R("results/raw_a30_s4"), SD, REF, c)
    ids = sorted(base.dec)
    L = ["# Session 4, item 1d — SharpDraft sanity anchor (4 reference traces, released drafter, full context, A30)",
         "", "| arm | τ | " + " | ".join(f"τ {b[0]}" for b in BINS) + " |", "|---|---|" + "---|" * len(BINS)]
    for a in (base, sd):
        L.append(f"| {a.label} | " + " | ".join(bin_cells(a, ids)) + " |")
    d = [tau(sd.stats(), ids) - tau(base.stats(), ids)]
    d += [tau(sd.stats(lo, hi), ids) - tau(base.stats(lo, hi), ids) for _, lo, hi in BINS]
    L.append("| Δ SharpDraft − baseline | " + " | ".join(f"{x:+.3f}" for x in d) + " |")
    L += ["", "| trace | baseline | SharpDraft |", "|---|---|---|"]
    for t in ids:
        s0, s1 = base.stats()[t], sd.stats()[t]
        L.append(f"| {t} | {s0[1] / s0[2]:.3f} | {s1[1] / s1[2]:.3f} |")
    L += [""] + scale_diag(sd, ids, "SharpDraft, full context")
    ok = d[0] > 0 and d[3] > d[1] and d[4] > d[1]
    L += ["", f"Rule (pre-registered): Δ overall {d[0]:+.3f} > 0, Δ(4–8K) {d[3]:+.3f} > Δ(0–2K) {d[1]:+.3f}, "
              f"Δ(≥8K) {d[4]:+.3f} > Δ(0–2K) → **{'PASS' if ok else 'FAIL'}**"]
    open(R("results/s4_anchor.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("ANCHOR", "PASS" if ok else "FAIL")


def refs_for(ids_all, c):
    rt, rc = R("results/raw_test"), R("results/raw_test_cal")
    tts = Arm("TTS stride 1", rt, TTS1, ids_all, c)
    cal = Arm("S2+cal", rc, S2, ids_all, c)
    return tts, cal, rt, rc


def item1(args):
    c, ids_all = cuts(), test_ids()
    tts, cal, rt, rc = refs_for(ids_all, c)
    base = Arm("baseline", rt, BASE, ids_all, c)
    s2 = Arm("S2", rt, S2, ids_all, c)
    ids = sorted(tts.dec)
    sd = Arm("SharpDraft", R("results/raw_test_s4"), SD, ids_all, c)
    lcb_ids = [t for t in ids_all if t.startswith("lcb")]
    sd_lcb = Arm("SD+S2+cal (LCB run)", R("results/raw_test_cal_s4"), SD_S2, lcb_ids, c)
    combo = Composite("SD+S2+cal", {"lcb": sd_lcb, "aime": cal})
    notes = ["SharpDraft (arXiv 2610.05106 Eq. 5/18; k 32, s 5, cap e^0.5, N0 2048 = the paper's fixed Qwen value, "
             "not tuned). SD = released drafter, full context. SD+S2+cal = frozen S2+cal checkpoint in the S2 "
             "layout: **LCB rounds from the new run (26 traces); AIME rounds are the S2+cal replay, identical by "
             "construction** (β̂ = 1 on every AIME round: N ≤ 1,965 ≤ N0; tests/test_sharpdraft.py). "
             "Update ms/round = 0 for both (one scalar per round)."]
    L, boot = heldout_tables([sd, combo], [tts, cal], ids, "Session 4, item 1 — SharpDraft (49 test traces, A30)", notes)
    L += ["", "## Further contrasts (paired 95% CI)", "", "| contrast | CI |", "|---|---|"]
    for a, b in ((sd, base), (sd, s2), (combo, sd)):
        sa, sb = a.stats(), b.stats()
        L.append(f"| {a.label} − {b.label} | {fmt_ci(*boot.ci(lambda d: tau(sa, d) - tau(sb, d)))} |")
    w, ci = reading(boot, sd, cal)
    w2, ci2 = reading(boot, combo, cal)
    w3, ci3 = reading(boot, combo, sd)
    L += ["", "## Readings (pre-registered)", "",
          f"- SD vs S2+cal: {fmt_ci(*ci)} → SD **{w}** S2+cal"
          + (" → our contribution is calibration + target-guided selection, not the window." if w == "within the CI of"
             else "."),
          f"- SD+S2+cal vs S2+cal {fmt_ci(*ci2)} ({w2}), vs SD {fmt_ci(*ci3)} ({w3}) → "
          + ("**complementary**" if ci2[1] > 0 and ci3[1] > 0 else "**not complementary** (rule: lower CI > 0 vs both)"),
          ""]
    L += scale_diag(sd, ids, "SD (full context)") + [""] + scale_diag(sd_lcb, sorted(sd_lcb.dec), "SD+S2+cal (S2 layout, LCB)")
    open(R("results/s4_sharpdraft.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def item2(args):
    c, ids_all = cuts(), test_ids()
    tts, cal, rt, rc = refs_for(ids_all, c)
    ids = sorted(tts.dec)
    d = R("results/raw_test_think")
    tb = Arm("thinking baseline", d, BASE, ids_all, c)
    ts2 = Arm("thinking + S2", d, S2, ids_all, c)
    base, s2 = Arm("baseline", rt, BASE, ids_all, c), Arm("S2", rt, S2, ids_all, c)
    notes = ["Drafter jgeuter/qwen3-4b-dpard-thinking-b16 @ 78baf299fc6dc793aab6c35ce44676a980e266d1 (3 layers, target "
             "feature layers [1, 17, 33]; released: 5 layers, [1, 9, 17, 25, 33]); no updates (0 ms/round)."]
    L, boot = heldout_tables([tb, ts2], [tts, cal], ids, "Session 4, item 2 — thinking-mode drafter (49 test traces, A30)",
                             notes)
    L += ["", "## Further contrasts (paired 95% CI)", "", "| contrast | CI |", "|---|---|"]
    for a, b in ((ts2, tb), (tb, base), (ts2, s2)):
        sa, sb = a.stats(), b.stats()
        L.append(f"| {a.label} − {b.label} | {fmt_ci(*boot.ci(lambda d_: tau(sa, d_) - tau(sb, d_)))} |")
    w, ci = reading(boot, tb, cal)
    L += ["", "## Reading (pre-registered)", "",
          f"- thinking baseline − S2+cal {fmt_ci(*ci)} ({w}) → "
          + ("**thinking baseline ≥ S2+cal: calibration's gain is available by retraining; lead with S2 and the hybrid.**"
             if ci[2] >= 0 else "**thinking baseline below S2+cal.**")]
    open(R("results/s4_thinking.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def item3(args):
    c, ids_all = cuts(), test_ids()
    tts, cal, rt, rc = refs_for(ids_all, c)
    ids = sorted(tts.dec)
    present = [t for t in ids_all if os.path.exists(os.path.join(R("results/raw_test_calfull"),
                                                                 f"{t}__baseline__default.jsonl"))]
    cf = Arm("cal-full", R("results/raw_test_calfull"), BASE, present, c)
    sub = sorted(cf.dec)
    base = Arm("baseline", rt, BASE, ids_all, c)
    partial = len(sub) < len(ids)
    notes = ["cal-full = released drafter calibrated in the full-context layout (same data, anchors, steps, lr as "
             "S2+cal), replayed with full context (no window). 0 ms/round."]
    if partial:
        notes.append(f"**PARTIAL: {len(sub)} of {len(ids)} test traces** (all contrasts on these traces only).")
    L, boot = heldout_tables([cf], [tts, cal], sub, "Session 4, item 3 — cal-full (A30)", notes)
    s2 = Arm("S2", rt, S2, ids_all, c)
    L += ["", "## Further contrasts (paired 95% CI)", "", "| contrast | CI |", "|---|---|"]
    for a, b in ((cf, base), (cf, s2), (cal, cf)):
        sa, sb = a.stats(), b.stats()
        L.append(f"| {a.label} − {b.label} | {fmt_ci(*boot.ci(lambda d: tau(sa, d) - tau(sb, d)))} |")
    for _, lo, hi in BINS:
        sa, sb = cal.stats(lo, hi), cf.stats(lo, hi)
        L.append(f"| S2+cal − cal-full, {_} | {fmt_ci(*boot.ci(lambda d: tau(sa, d) - tau(sb, d)))} |")
    w, ci = reading(boot, cf, cal)
    L += ["", "## Reading (pre-registered)", "",
          f"- cal-full − S2+cal {fmt_ci(*ci)} ({w})"
          + (" → **the S2 layout adds nothing on top of calibration.**" if w == "within the CI of" else ".")]
    open(R("results/s4_calfull.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anchor", "item1", "item2", "item3"])
    a = ap.parse_args()
    {"anchor": anchor, "item1": item1, "item2": item2, "item3": item3}[a.cmd](a)


if __name__ == "__main__":
    main()
