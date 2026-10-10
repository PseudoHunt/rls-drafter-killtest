"""Screen report: KL-bias × 3 and KL-linear × 3 on the 4 TTS-reference traces -> results/screen.md.

R = (τ_arm − τ_base,A30) / gain_TTS, with gain_TTS = τ_TTS − τ_base from the old GPU (results/raw, lr = 3e-5),
because TTS was not re-run on this GPU. Decision rule pre-registered in RUNLOG.md (best arm R ≥ 0.5 promising,
R < 0.25 stop, else report and ask). τ: flagged traces truncated (none of these 4 is flagged), pooled over
rounds per dataset, mean of AIME and LCB.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import load_cuts, load_raw, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = ["aime24_II_14", "aime24_I_10", "lcb_abc393_d", "lcb_arc191_d"]
BASE, TTS = ("baseline", "default"), ("tts", "lr=3e-05")


STOP = "stop this line"


def band(r):
    return "promising" if r >= 0.5 else (STOP if r < 0.25 else "in between: report and ask")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results/raw_a30"))
    ap.add_argument("--old-raw", default=os.path.join(ROOT, "results/raw"))
    ap.add_argument("--cuts", default=os.path.join(ROOT, "results/loop_cuts.csv"))
    ap.add_argument("--ids", nargs="*", default=REF)
    ap.add_argument("--tts-gain", type=float, default=None, help="override (dry runs); default: from --old-raw")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/screen.md"))
    ap.add_argument("--gpu", default="A30")
    ap.add_argument("--arms", nargs="+", default=["klbias", "kllinear"])
    ap.add_argument("--title", default="Screen: KL-bias and KL-linear vs baseline (4 TTS-reference traces)")
    ap.add_argument("--stop-label", default="stop this line")
    args = ap.parse_args()
    global STOP
    STOP = args.stop_label

    cuts = load_cuts(args.cuts) if os.path.exists(args.cuts) else {}
    pick = lambda runs: [r for r in runs if r["header"]["trace_id"] in args.ids]  # noqa: E731
    tab = table(pick(load_raw(args.raw)), "trunc", cuts)
    tau_b = tab[BASE]["mean"]["tau"]
    if args.tts_gain is None:
        old = table(pick(load_raw(args.old_raw)), "trunc", cuts)
        ob, ot = old[BASE]["mean"]["tau"], old[TTS]["mean"]["tau"]
        gain = ot - ob
        src = f"old GPU: τ_TTS(lr=3e-5) − τ_base = {ot:.3f} − {ob:.3f} = {gain:.3f}"
        drift = (tau_b - ob) / ob
        drift_line = (f"{args.gpu} baseline τ = {tau_b:.3f} vs old-GPU {ob:.3f} ({100 * drift:+.2f}%)"
                      + (" — **differs by more than 1%: flag**" if abs(drift) > 0.01 else " — within 1%."))
    else:
        gain, src, drift_line = args.tts_gain, f"--tts-gain {args.tts_gain} (dry run)", "dry run"

    L = [f"# {args.title}", "",
         f"Traces: {', '.join(args.ids)}. GPU: {args.gpu}. τ = committed tokens/round, flagged traces truncated, "
         "pooled per dataset, mean of AIME and LCB.", "",
         f"- τ_base ({args.gpu}) = {tau_b:.3f}. {drift_line}",
         f"- gain_TTS = {gain:.3f} ({src}); R = (τ − τ_base,{args.gpu}) / gain_TTS.",
         f"- Thresholds: R = 0.5 ↔ τ = {tau_b + 0.5 * gain:.3f}; R = 0.25 ↔ τ = {tau_b + 0.25 * gain:.3f}.",
         "- Tuned on the same 4 traces, so the best arm's R is optimistic.", "",
         "| method | config | τ AIME | τ LCB | τ mean | R | τ <2K | τ ≥8K | update ms/round | draft ms/round | state MB |"
         + (" KL-linear rounds with ‖Az̃‖/‖z‖ > 0.2 (max) |" if "kllinear" in args.arms else ""),
         "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if "kllinear" in args.arms else "")]
    keys = [BASE] + sorted(k for k in tab if k[0] in args.arms)
    best = None
    for k in keys:
        r = tab[k]
        m = r["mean"]
        R = (m["tau"] - tau_b) / gain
        flag = "–"
        if k[0] == "kllinear":
            runs = [x for x in pick(load_raw(args.raw)) if (x["header"]["method"], x["header"]["config"]) == k]
            n = sum(x["header"]["diagnostics"].get("n_rounds_rel_gt_flag", 0) for x in runs)
            mx = max(x["header"]["diagnostics"].get("max_rel", 0.0) for x in runs)
            flag = f"{n} ({mx:.3f})" + (" **FLAG**" if n else "")
        L.append(f"| {k[0]} | {k[1]} | {r['aime']['tau']:.3f} | {r['lcb']['tau']:.3f} | {m['tau']:.3f} | "
                 f"{R:+.3f} | {m['tau_lt2k']:.3f} | {m['tau_ge8k']:.3f} | {m['update_ms']:.2f} | {m['draft_ms']:.2f} | "
                 f"{m['state_mb']:.2f} |" + (f" {flag} |" if "kllinear" in args.arms else ""))
        if k != BASE and (best is None or R > best[1]):
            best = (k, R, m["tau"])
    L += ["", "Timings were measured under concurrent workers (MPS), not isolated; τ <2K / ≥8K = output positions."
          + (" State MB excludes the shared fp32 copy of the lm_head weight (1.45 GiB on Qwen3-4B) that both KL arms "
             "use for W." if "kllinear" in args.arms else ""), ""]
    if best:
        L += ["## Decision (pre-registered, RUNLOG.md)", "",
              f"Best arm: {best[0][0]} {best[0][1]}, τ = {best[2]:.3f}, R = {best[1]:+.3f} → **{band(best[1])}**.", ""]
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"SCREEN_DECISION {band(best[1]) if best else 'none'}")


if __name__ == "__main__":
    main()
