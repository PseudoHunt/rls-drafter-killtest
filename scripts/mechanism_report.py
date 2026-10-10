"""Mechanism + selection screen report (NOTES.md §10) -> results/mechanism.md.

τ, R = (τ − τ_base,A30)/gain_TTS, τ <2K / ≥8K, per-dataset τ, drafting ms/round, Δτ vs the frozen window
(W = 1024, keep prompt), and the three pre-registered readings (RUNLOG.md).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import load_cuts, load_raw, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = ["aime24_II_14", "aime24_I_10", "lcb_abc393_d", "lcb_arc191_d"]
BASE, TTS = ("baseline", "default"), ("tts", "lr=3e-05")
WIN = ("window", "W=1024_keep=prompt")
ROWS = [(BASE, "baseline"), (WIN, "window (frozen)"), (("winorig", "W=1024_keep=prompt"), "B: window, original positions"),
        (("capctx", "cap=4096"), "E: all keys, capped distance"), (("topk", "k=1024"), "C: top-k, original positions"),
        (("topkcap", "cap=4096_k=1024"), "C′: top-k, capped positions"),
        (("topkshared", "cap=4096_k=1024"), "D: top-k shared over block, capped")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results/raw_a30_window"))
    ap.add_argument("--old-raw", default=os.path.join(ROOT, "results/raw"))
    ap.add_argument("--cuts", default=os.path.join(ROOT, "results/loop_cuts.csv"))
    ap.add_argument("--tts-gain", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/mechanism.md"))
    ap.add_argument("--gpu", default="A30")
    args = ap.parse_args()
    cuts = load_cuts(args.cuts) if os.path.exists(args.cuts) else {}
    pick = lambda runs: [r for r in runs if r["header"]["trace_id"] in REF]  # noqa: E731
    tab = table(pick(load_raw(args.raw)), "trunc", cuts)
    if args.tts_gain is None:
        old = table(pick(load_raw(args.old_raw)), "trunc", cuts)
        gain = old[TTS]["mean"]["tau"] - old[BASE]["mean"]["tau"]
    else:
        gain = args.tts_gain
    tb, tw = tab[BASE]["mean"]["tau"], tab[WIN]["mean"]["tau"]
    L = ["# Mechanism + selection screen (4 TTS-reference traces)", "",
         f"GPU {args.gpu}. τ_base = {tb:.3f}; gain_TTS = {gain:.3f} (old GPU); R = (τ − τ_base)/gain_TTS. Key budget "
         "1024 for selection. τ: flagged traces truncated (none here), pooled per dataset, mean of AIME and LCB; "
         "τ <2K / ≥8K by output position. Draft ms/round measured under concurrent workers (MPS): baseline and window "
         "rows from the window run (3 workers); new rows mostly with 2 workers (a 3-worker attempt hit CUDA OOM; 6 jobs "
         "finished under 3 workers were kept), so timings are indicative only.", "",
         "| config | τ mean | R | Δτ vs window | τ AIME | τ LCB | τ <2K | τ ≥8K | draft ms/round |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, label in ROWS:
        if key not in tab:
            L.append(f"| {label} | missing | | | | | | | |")
            continue
        r, m = tab[key], tab[key]["mean"]
        L.append(f"| {label} | {m['tau']:.3f} | {(m['tau'] - tb) / gain:+.3f} | {m['tau'] - tw:+.3f} | "
                 f"{r['aime']['tau']:.3f} | {r['lcb']['tau']:.3f} | {m['tau_lt2k']:.3f} | {m['tau_ge8k']:.3f} | "
                 f"{m['draft_ms']:.2f} |")
    g = lambda k: tab[k]["mean"]["tau"] if k in tab else float("nan")  # noqa: E731
    tE = g(("capctx", "cap=4096"))
    sel = {"C′": g(("topkcap", "cap=4096_k=1024")), "D": g(("topkshared", "cap=4096_k=1024"))}
    best_sel = max(sel, key=sel.get)
    L += ["", "## Pre-registered readings (RUNLOG.md)", ""]
    frac = (tE - tb) / (tw - tb)
    L.append(f"- E vs window: Δτ = {tE - tw:+.3f}, E recovers {100 * frac:.0f}% of the window's gain → " + (
        "**within 0.1 τ: distance is the main cause.**" if abs(tE - tw) <= 0.1 else
        ("**E near baseline (within 0.1 τ) while the window is far above: key count or distant noise is the cause.**"
         if abs(tE - tb) <= 0.1 else
         "**neither pre-registered reading applies cleanly: capping distance recovers part of the gain; the rest "
         "needs the distant keys removed (key count / distant noise).**")))
    d = sel[best_sel] - tw
    L.append(f"- Best of C′/D: {best_sel}, Δτ vs window = {d:+.3f} → " + (
        "**beats the window by ≥ 0.1 τ: selection adds value; CrossCov-U next.**" if d >= 0.1 else
        "**does not beat the window by ≥ 0.1 τ: no selection gain shown.**"))
    best_any = max((g(k), lab) for k, lab in ROWS if k[0] in ("capctx", "topkcap", "topkshared"))
    L.append(f"- Held-out candidate besides the frozen window: "
             + (f"{best_any[1]} (τ {best_any[0]:.3f}, Δ {best_any[0] - tw:+.3f})" if best_any[0] - tw >= 0.1
                else "none (no C′/D/E config beats the window by ≥ 0.1 τ)."))
    L.append("")
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
