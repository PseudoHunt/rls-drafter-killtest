"""Interim dev table at pause time (TTS grid incomplete): all completed configs (truncated variant) and the
4-trace reference comparison against TTS lr=3e-5. Writes results/dev_interim.md."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import load_cuts, load_raw, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = ["aime24_II_14", "aime24_I_10", "lcb_abc393_d", "lcb_arc191_d"]


def f(x):
    return "–" if x != x else f"{x:.3f}"


def rows(tab, title, note):
    L = [f"## {title}", "", note, "",
         "| method | config | τ AIME (n) | τ LCB (n) | τ mean | τ <2K | τ ≥8K | draft ms | update ms | solve ms | state MB |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["baseline", "bias", "rls", "tts"]
    for k in sorted(tab, key=lambda k: (order.index(k[0]), k[1])):
        r = tab[k]
        a, l = r.get("aime"), r.get("lcb")
        m = r.get("mean")
        cell = lambda d: f"{f(d['tau'])} ({d['n_traces']})" if d else "–"  # noqa: E731
        src = m or a or l
        L.append(f"| {k[0]} | {k[1]} | {cell(a)} | {cell(l)} | {f(m['tau']) if m else '–'} | "
                 f"{f(m['tau_lt2k']) if m else '–'} | {f(m['tau_ge8k']) if m else '–'} | {src['draft_ms']:.2f} | "
                 f"{src['update_ms']:.2f} | {src['solve_ms']:.2f} | {src['state_mb']:.0f} |")
    return L + [""]


def main():
    cuts = load_cuts(os.path.join(ROOT, "results/loop_cuts.csv"))
    runs = load_raw(os.path.join(ROOT, "results/raw"), "dev")
    full = table(runs, "trunc", cuts)
    L = ["# Dev results at pause (interim; Stage 3 stopped after the RLS futility result)", "",
         "Decision variant: flagged traces truncated at loop onset (§11 A2, §12.4); τ pooled over rounds per dataset; "
         "mean = average of AIME and LCB. Timings were measured under MPS concurrency (not isolated).", ""]
    L += rows(full, "All 12 dev traces (11 after the drop)", "TTS rows cover only the dev traces that finished before the grid was cancelled (n in brackets).")
    ref = table([r for r in runs if r["header"]["trace_id"] in REF], "trunc", cuts)
    L += rows(ref, "TTS reference set: 4 dev traces (" + ", ".join(REF) + ")", "All configs on exactly these 4 traces.")
    b, t = ref[("baseline", "default")]["mean"]["tau"], ref[("tts", "lr=3e-05")]["mean"]["tau"]
    best = {}
    for (m, c), r in ref.items():
        if m in ("bias", "rls") and (m not in best or r["mean"]["tau"] > ref[(m, best[m])]["mean"]["tau"]):
            best[m] = c
    L += ["## R on the reference set", "", f"τ_base = {b:.3f}, τ_TTS(lr=3e-5) = {t:.3f}.", "",
          "| arm | config | τ | R |", "|---|---|---|---|"]
    for m, c in best.items():
        tau = ref[(m, c)]["mean"]["tau"]
        L.append(f"| {'hidden-residual bias' if m == 'bias' else 'RLS'} (best on these 4) | {c} | {tau:.3f} | {(tau - b) / (t - b):.3f} |")
    best_full = max(((r["mean"]["tau"], c) for (m, c), r in full.items() if m == "rls" and "mean" in r))
    bf, tf_ = full[("baseline", "default")]["mean"]["tau"], best_full[0]
    L += ["", f"Futility (§12.6): best RLS on all dev traces = {best_full[1]} with τ = {tf_:.3f} < τ_base = {bf:.3f} → dev R < 0 "
          "for any TTS that beats the baseline → **STOP**.", ""]
    open(os.path.join(ROOT, "results/dev_interim.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
