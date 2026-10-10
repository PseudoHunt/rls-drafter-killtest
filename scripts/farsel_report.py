"""T0b target-guided selection screen report (NOTES.md §11) -> results/farsel.md.

Per config: τ, R, Δτ vs the frozen window, τ AIME, τ LCB, τ <2K, τ ≥8K, draft ms/round, mean visible context keys,
and the pre-registered rule: target-guided selection adds value only if the best of S1/S2/S3 beats the window by
≥ 0.1 τ AND beats both C1 and C2.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import load_cuts, load_raw, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = ["aime24_II_14", "aime24_I_10", "lcb_abc393_d", "lcb_arc191_d"]
BASE, TTS = ("baseline", "default"), ("tts", "lr=3e-05")
WIN = ("window", "W=1024_keep=prompt")
S = {"S1": ("farsel", "W=1024_keep=prompt_n=16_score=last"), "S2": ("farsel", "W=1024_keep=prompt_n=16_score=mid3"),
     "S3": ("farsel", "W=512_keep=prompt_n=16_score=last")}
C = {"C1": ("window", "W=1536_keep=prompt"), "C2": ("farsel", "W=1024_keep=prompt_n=16_score=random_seed=0")}
LABEL = {BASE: "baseline", WIN: "window W 1024 + prompt (frozen)",
         S["S1"]: "S1: W 1024 + prompt + 16 far chunks, target last layer",
         S["S2"]: "S2: W 1024 + prompt + 16 far chunks, target layers 17/25/33",
         S["S3"]: "S3: W 512 + prompt + 16 far chunks, last layer (window budget)",
         C["C1"]: "C1: window W 1536 + prompt (S1 budget)",
         C["C2"]: "C2: W 1024 + prompt + 16 random far chunks (seed 0)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results/raw_a30_window"))
    ap.add_argument("--old-raw", default=os.path.join(ROOT, "results/raw"))
    ap.add_argument("--cuts", default=os.path.join(ROOT, "results/loop_cuts.csv"))
    ap.add_argument("--tts-gain", type=float, default=None)
    ap.add_argument("--row-cost", default=None, help="markdown table from scripts/attn_row_cost.py")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/farsel.md"))
    ap.add_argument("--gpu", default="A30")
    args = ap.parse_args()
    cuts = load_cuts(args.cuts) if os.path.exists(args.cuts) else {}
    runs = [r for r in load_raw(args.raw) if r["header"]["trace_id"] in REF]
    tab = table(runs, "trunc", cuts)
    if args.tts_gain is None:
        old = table([r for r in load_raw(args.old_raw) if r["header"]["trace_id"] in REF], "trunc", cuts)
        gain = old[TTS]["mean"]["tau"] - old[BASE]["mean"]["tau"]
    else:
        gain = args.tts_gain
    tb, tw = tab[BASE]["mean"]["tau"], tab[WIN]["mean"]["tau"]
    ctx = {}
    for r in runs:
        k = (r["header"]["method"], r["header"]["config"])
        ctx.setdefault(k, []).extend(x.get("ctx_len", 0) for x in r["rounds"])
    L = ["# T0b: target-guided far-context selection (4 TTS-reference traces, A30)", "",
         f"τ_base = {tb:.3f}; window (frozen) = {tw:.3f}; gain_TTS = {gain:.3f} (old GPU); R = (τ − τ_base)/gain_TTS. "
         "Far chunks = aligned 32-token chunks wholly older than the window (and after the prompt), re-selected every "
         "round from the target's head-summed attention row at the last committed position t'−1 (guarded), kept in "
         "original order, renumbered contiguously. Draft ms/round under 2 concurrent workers (MPS), indicative; "
         "baseline and window rows from the earlier window run (3 workers).", "",
         "| config | τ | R | Δτ vs window | τ AIME | τ LCB | τ <2K | τ ≥8K | draft ms/round | mean visible keys |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for key in [BASE, WIN, S["S1"], S["S2"], S["S3"], C["C1"], C["C2"]]:
        if key not in tab:
            L.append(f"| {LABEL[key]} | missing | | | | | | | | |")
            continue
        r, m = tab[key], tab[key]["mean"]
        mk = sum(ctx[key]) / max(len(ctx[key]), 1) if any(ctx.get(key, [])) else float("nan")
        L.append(f"| {LABEL[key]} | {m['tau']:.3f} | {(m['tau'] - tb) / gain:+.3f} | {m['tau'] - tw:+.3f} | "
                 f"{r['aime']['tau']:.3f} | {r['lcb']['tau']:.3f} | {m['tau_lt2k']:.3f} | {m['tau_ge8k']:.3f} | "
                 f"{m['draft_ms']:.2f} | {mk:.0f} |")
    g = lambda k: tab[k]["mean"]["tau"] if k in tab else float("nan")  # noqa: E731
    best = max(S, key=lambda n: g(S[n]))
    tbest, c1, c2 = g(S[best]), g(C["C1"]), g(C["C2"])
    adds = tbest - tw >= 0.1 and tbest > c1 and tbest > c2
    L += ["", "## Pre-registered rule (RUNLOG.md)", "",
          f"Best of S1/S2/S3: **{best}**, τ = {tbest:.3f}: vs window {tbest - tw:+.3f} (need ≥ +0.1), vs C1 "
          f"{tbest - c1:+.3f}, vs C2 {tbest - c2:+.3f} (need > 0 for both).", "",
          ("**Target-guided selection adds value → freeze " + best + " and add it as an arm in T1 and T2.**") if adds else
          "**Target-guided selection does not add value under the rule → proceed with the plan unchanged.**", ""]
    if args.row_cost and os.path.exists(args.row_cost):
        L += ["## Extra target-side cost of the selection signal (A30 microbenchmark, scripts/attn_row_cost.py)", "",
              "One attention row at t'−1 per round from the resident KV cache, head-summed, chunk-reduced, top-16. "
              "Unfused PyTorch (fp32 copy of K), so an upper bound; the replay itself precomputes all rows in the "
              "teacher-forced pass, which is not a deployment cost.", "",
              open(args.row_cost).read().split("\n", 1)[1].strip(), ""]
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))
    print("FARSEL_DECISION", "adds_value" if adds else "no_value", best)


if __name__ == "__main__":
    main()
