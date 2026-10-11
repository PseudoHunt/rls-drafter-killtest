"""Session-3 reports (pre-registered in RUNLOG.md, 2026-10-11): Step 1 (batched pooled update), Step 4 (screen on the
4 reference traces), Step 5 (held-out), Step 6 (refresh vs stale).

  python scripts/s3_report.py step1            -> results/batched.md
  python scripts/s3_report.py screen --cal-dir results/raw_a30_cal_lr1e-05   -> results/s3_screen.md (+ readings)
  python scripts/s3_report.py heldout --arms S2+cal full                     -> results/s3_heldout.md
  python scripts/s3_report.py lrpick           -> prints S2+cal τ per lr on the 4 reference traces

τ: decision variant (flagged traces truncated at loop onset, dropped excluded), mean over AIME and LCB of pooled τ;
bins by round start output position. CIs: scripts/heldout_report.Boot (paired, stratified, B = 10,000, seed 0).
Amortized update ms/round = mean of per-round update_ms over all rounds of the full-trace replays (as heldout.md).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import DATASETS, load_cuts, select  # noqa: E402
from heldout_report import Boot, fmt_ci  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: os.path.join(ROOT, p)  # noqa: E731
REF = ["aime24_II_14", "aime24_I_10", "lcb_abc393_d", "lcb_arc191_d"]
T2 = open(R("configs/t2_traces.txt")).read().split()
S2 = ("farsel", "W=1024_keep=prompt_n=16_score=mid3")
TTS1 = ("tts", "lr=1e-05")
HYB = ("tts", "ctx=s2_lr=1e-05_pool=True_stride=10")
B10 = ("tts", "batched=True_ctx=s2_lr=1e-05_pool=True_stride=10")
B20 = ("tts", "batched=True_ctx=s2_lr=1e-05_pool=True_stride=20")
WARM = ("tts", "batched=True_ctx=s2_lr=1e-05_pool=True_stride=10_warm=2048")
REFRESH = ("tts", "lr=1e-05_refresh=True")
BINS = [("0–2K", 0, 2048), ("2–4K", 2048, 4096), ("4–8K", 4096, 8192), ("≥8K", 8192, None)]
LOW = ("0–4K", 0, 4096)


def path_for(dirpath, tr, key):
    return os.path.join(dirpath, f"{tr}__{key[0]}__{key[1]}.jsonl")


def load_runs(dirpath, traces, key, required=True):
    runs = []
    for tr in traces:
        p = path_for(dirpath, tr, key)
        if not os.path.exists(p):
            if required:
                raise FileNotFoundError(p)
            continue
        lines = [json.loads(x) for x in open(p)]
        runs.append({"header": lines[0], "rounds": [r for r in lines if r.get("type") == "round"]})
    return runs


class Arm:
    def __init__(self, label, dirpath, key, traces, cuts):
        self.label, self.key = label, key
        self.full = load_runs(dirpath, traces, key)
        self.dec = {r["header"]["trace_id"]: r for r in select(self.full, "trunc", cuts)}
        self.ds = {t: r["header"]["dataset"] for t, r in self.dec.items()}

    def stats(self, lo=None, hi=None):
        out = {}
        for t, r in self.dec.items():
            rs = [x for x in r["rounds"] if (lo is None or x["pos"] >= lo) and (hi is None or x["pos"] < hi)]
            out[t] = (self.ds[t], sum(x["tau"] for x in rs), len(rs))
        return out

    def update_ms(self):
        rows = [x for r in self.full for x in r["rounds"]]
        return sum(x["update_ms"] for x in rows) / len(rows), len(rows)

    def draft_ms(self):
        rows = [x for r in self.full for x in r["rounds"]]
        return sum(x["draft_ms"] for x in rows) / len(rows)


def tau(stats, ids):
    by = {ds: [0, 0] for ds in DATASETS}
    for i in ids:
        ds, tok, n = stats[i]
        by[ds][0] += tok
        by[ds][1] += n
    vals = [by[ds][0] / by[ds][1] for ds in DATASETS if by[ds][1]]
    return sum(vals) / len(vals) if vals else float("nan")


def per_trace_tau(stats, i):
    _, tok, n = stats[i]
    return tok / n if n else float("nan")


def tau_table(arms, ids):
    hdr = "| arm | traces | τ | " + " | ".join(f"τ {b[0]}" for b in BINS) + " | τ 0–4K | update ms/round | rounds |"
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for a in arms:
        s = a.stats()
        cells = [f"{tau(s, ids):.3f}"] + [f"{tau(a.stats(lo, hi), ids):.3f}" for _, lo, hi in BINS]
        cells.append(f"{tau(a.stats(LOW[1], LOW[2]), ids):.3f}")
        u, n = a.update_ms()
        lines.append(f"| {a.label} | {len(ids)} | " + " | ".join(cells) + f" | {u:.2f} | {n} |")
    return "\n".join(lines)


def cuts():
    return load_cuts(R("results/loop_cuts.csv"))


def step1(args):
    c = cuts()
    rt = R("results/raw_test")
    arms = [Arm("TTS stride 1 (existing)", rt, TTS1, T2, c), Arm("hybrid, per-round pooled (existing)", rt, HYB, T2, c),
            Arm("batched pooled stride 10", rt, B10, T2, c)]
    ids = sorted(arms[0].dec)
    have20 = [t for t in T2 if os.path.exists(path_for(rt, t, B20))]
    note20 = ""
    if len(have20) == len(T2):
        arms.append(Arm("batched pooled stride 20 (exploratory)", rt, B20, T2, c))
    elif have20:
        note20 = (f"Stride 20 (exploratory) ran on {len(have20)}/16 traces only (budget): "
                  + ", ".join(have20) + "; table on that subset below.")
    tts_u = arms[0].update_ms()[0]
    t_h, t_b = tau(arms[1].stats(), ids), tau(arms[2].stats(), ids)
    lines = ["# Step 1 — exact batched pooled update (16 T2 traces, A30)", "",
             "Pre-registered in RUNLOG.md (session 3). Decision variant; one TTS process alone on the GPU for every row.",
             "", tau_table(arms, ids), "",
             "| variant | amortized update ms/round | ratio to TTS stride 1 | ≤ 10%? | draft ms/round |", "|---|---|---|---|---|"]
    for a in arms:
        u = a.update_ms()[0]
        lines.append(f"| {a.label} | {u:.2f} | {100 * u / tts_u:.1f}% | {'yes' if u <= 0.1 * tts_u else 'no'} | "
                     f"{a.draft_ms():.2f} |")
    boot = Boot(ids, arms[0].ds, 10000)
    s0, s2 = arms[0].stats(), arms[2].stats()
    lines += ["", f"- **τ check:** batched stride 10 τ = {t_b:.3f} vs per-round pooled hybrid {t_h:.3f} "
              f"(|Δ| = {abs(t_b - 4.429):.3f} vs 4.429; need ≤ 0.02) → **{'PASS' if abs(t_b - 4.429) <= 0.02 else 'FAIL'}**",
              f"- batched stride 10 − TTS stride 1: {fmt_ci(*boot.ci(lambda d: tau(s2, d) - tau(s0, d)))}",
              f"- cost criterion (≤ 10% of TTS stride 1 = {0.1 * tts_u:.2f} ms): batched stride 10 "
              f"{arms[2].update_ms()[0]:.2f} ms → **{'met' if arms[2].update_ms()[0] <= 0.1 * tts_u else 'not met'}**"]
    hy = {t: per_trace_tau(arms[1].stats(), t) for t in ids}
    bt = {t: per_trace_tau(s2, t) for t in ids}
    lines += ["", "| trace | hybrid τ | batched s10 τ | Δ |", "|---|---|---|---|"]
    lines += [f"| {t} | {hy[t]:.3f} | {bt[t]:.3f} | {bt[t] - hy[t]:+.3f} |" for t in ids]
    if note20:
        sub = [t for t in ids if t in have20]
        a20 = Arm("batched pooled stride 20 (exploratory)", rt, B20, sub, c)
        sub_arms = [Arm(a.label, rt, a.key, sub, c) for a in arms] + [a20]
        lines += ["", note20, "", tau_table(sub_arms, sorted(a20.dec))]
    open(R("results/batched.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def lrpick(args):
    c = cuts()
    for d in sorted(x for x in os.listdir(R("results")) if x.startswith("raw_a30_cal_lr")):
        a = Arm(d, R("results/" + d), S2, REF, c)
        print(f"{d}: S2+cal τ = {tau(a.stats(), REF):.3f}  τ 0–4K = {tau(a.stats(0, 4096), REF):.3f}")


def screen(args):
    c = cuts()
    s3, cal = R("results/raw_a30_s3"), R(args.cal_dir)
    arms = [Arm("S2", R("results/raw_a30_window"), S2, REF, c), Arm("TTS stride 1", s3, TTS1, REF, c),
            Arm("S2+cal (static)", cal, S2, REF, c), Arm("full = S2+cal + batched TTS s10", cal, B10, REF, c)]
    if os.path.exists(path_for(s3, REF[0], WARM)):
        arms.append(Arm("S2 + TTS s1 first 2K, then batched s10", s3, WARM, REF, c))
    ids = REF
    S, T, C, F = (a.stats for a in arms[:4])
    lo_s2, lo_t, lo_c = (tau(f(0, 4096), ids) for f in (S, T, C))
    deficit = lo_t - lo_s2
    closed = (lo_c - lo_s2) / deficit if deficit > 0 else float("nan")
    t_t, t_c, t_f = tau(T(), ids), tau(C(), ids), tau(F(), ids)
    wins = sum(per_trace_tau(F(), i) > per_trace_tau(T(), i) for i in ids)
    lines = ["# Session 3, Step 4 — screen (4 reference traces, A30)", "", f"cal checkpoint: `{args.cal_dir}`", "",
             tau_table(arms, ids), "",
             "| trace | " + " | ".join(a.label for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for i in ids:
        lines.append(f"| {i} | " + " | ".join(f"{per_trace_tau(a.stats(), i):.3f}" for a in arms) + " |")
    mm = deficit > 0 and closed >= 0.5
    pf = t_f >= t_t + 0.10 and wins >= 3
    pc = t_c >= t_t
    lines += ["", "Readings (pre-registered):",
              f"- Mode mismatch: S2 0–4K τ {lo_s2:.3f}, TTS {lo_t:.3f} (deficit {deficit:+.3f}), S2+cal {lo_c:.3f} → "
              f"closes {100 * closed:.0f}% (need ≥ 50%) → **{'supported' if mm else 'NOT supported'}**",
              f"- Promote full: τ_full {t_f:.3f} vs τ_TTS + 0.10 = {t_t + 0.10:.3f}; full > TTS on {wins}/4 traces → "
              f"**{'PROMOTE' if pf else 'no'}**",
              f"- Promote S2+cal: τ_S2cal {t_c:.3f} vs τ_TTS {t_t:.3f} → **{'PROMOTE' if pc else 'no'}**"]
    pw = None
    if len(arms) > 4:
        W_ = arms[4].stats
        t_w = tau(W_(), ids)
        ww = sum(per_trace_tau(W_(), i) > per_trace_tau(T(), i) for i in ids)
        pw = t_w >= t_t + 0.10 and ww >= 3
        lines.append(f"- Promote warm-start arm: τ {t_w:.3f} vs {t_t + 0.10:.3f}; beats TTS on {ww}/4 → "
                     f"**{'PROMOTE' if pw else 'no'}**")
    lines.append("- Selection optimism: the S2+cal lr was chosen on these 4 traces (TTS lr also dev-tuned).")
    open(R("results/s3_screen.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("READINGS " + json.dumps({"mode_mismatch": bool(mm), "promote_full": bool(pf), "promote_s2cal": bool(pc),
                                    "promote_warm": pw}))


def heldout(args):
    c = cuts()
    rt, rc = R("results/raw_test"), R("results/raw_test_cal")
    test_ids = sorted(json.loads(x)["trace_id"] for x in open(R("traces/traces.jsonl"))
                      if json.loads(x)["split"] == "test")
    tts, s2 = Arm("TTS stride 1", rt, TTS1, test_ids, c), Arm("S2", rt, S2, test_ids, c)
    spec = {"S2+cal": (rc, S2), "full": (rc, B10), "warm": (R("results/raw_test_s3"), WARM)}
    arms = [Arm(a, *spec[a], test_ids, c) for a in args.arms]
    ids = sorted(tts.dec)
    boot = Boot(ids, tts.ds, 10000)
    tts_u = tts.update_ms()[0]
    lines = ["# Session 3, Step 5 — held-out (49 test traces, A30)", "", tau_table([s2, tts] + arms, ids), "",
             "| arm | arm − TTS stride 1 | arm − S2 | beats TTS? | update ms/round (ratio) | cheaper? |",
             "|---|---|---|---|---|---|"]
    caveat = "" if args.refresh_checked else " (stale-cache TTS; refresh check not run)"
    for a in arms:
        sa, st, ss = a.stats(), tts.stats(), s2.stats()
        d_t = boot.ci(lambda d: tau(sa, d) - tau(st, d))
        d_s = boot.ci(lambda d: tau(sa, d) - tau(ss, d))
        u = 0.0 if a.label == "S2+cal" else a.update_ms()[0]
        lines.append(f"| {a.label} | {fmt_ci(*d_t)} | {fmt_ci(*d_s)} | {'YES' if d_t[1] > 0 else 'no'}{caveat} | "
                     f"{u:.2f} ({100 * u / tts_u:.1f}%) | {'YES' if u <= 0.1 * tts_u else 'no'} |")
    lines.append(f"\nTTS stride 1 amortized update ms/round on these traces: {tts_u:.2f}.")
    open(R("results/s3_heldout.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["step1", "screen", "heldout", "lrpick"])
    ap.add_argument("--cal-dir", default=None)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--refresh-checked", action="store_true")
    a = ap.parse_args()
    {"step1": step1, "screen": screen, "heldout": heldout, "lrpick": lrpick}[a.cmd](a)


if __name__ == "__main__":
    main()
