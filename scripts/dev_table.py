"""Gate C: dev table for every config, best config per method, dev R, futility check; writes configs/frozen.yaml
and results/dev_table.md. Decision metric = truncation variant (Amendments 2 + §12.4)."""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import load_cuts, load_raw, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ("aime", "lcb", "mean")


def fmt(x):
    return "–" if x != x else f"{x:.3f}"


def pick(tab):
    best = {}
    for m in ("baseline", "bias", "rls"):
        cands = [(res["mean"]["tau"], cfg) for (mm, cfg), res in tab.items() if mm == m and "mean" in res]
        best[m] = max(cands)[1]
    stale = {cfg: res["mean"]["tau"] for (m, cfg), res in tab.items() if m == "tts" and "refresh" not in cfg}
    lr = max(stale, key=stale.get)
    ref_key = ("tts", f"{lr}_refresh=True")
    refresh_used = ref_key in tab and tab[ref_key]["mean"]["tau"] > 1.05 * stale[lr]
    if refresh_used:
        refs = {cfg: res["mean"]["tau"] for (m, cfg), res in tab.items() if m == "tts" and "refresh" in cfg}
        best["tts"] = max(refs, key=refs.get)
    else:
        best["tts"] = lr
    return best, refresh_used


def R(tab, key, base, tts, metric="tau"):
    b, t = tab[base]["mean"][metric], tab[tts]["mean"][metric]
    return (tab[key]["mean"][metric] - b) / (t - b) if t != b else float("nan")


def main():
    cuts = load_cuts(os.path.join(ROOT, "results/loop_cuts.csv"))
    runs = load_raw(os.path.join(ROOT, "results/raw"), "dev")
    tabs = {v: table(runs, v, cuts) for v in ("trunc", "exclude", "loops")}
    tab = tabs["trunc"]
    best, refresh_used = pick(tab)
    base, tts = ("baseline", best["baseline"]), ("tts", best["tts"])
    lines = ["# Dev table (Stage 3)", "",
             "Decision variant: flagged traces truncated at loop onset (KILL_TEST_BRIEF §11 A2, §12.4). "
             "τ pooled over rounds per dataset; mean = average of AIME and LCB. R uses the selected baseline and TTS.", ""]
    for v, title in (("trunc", "Truncated (decision)"), ("exclude", "Exclusion (flagged dropped)"), ("loops", "Loops included")):
        t = tabs[v]
        lines += [f"## {title}", "",
                  "| method | config | τ AIME | τ LCB | τ mean | τ<2K mean | τ≥8K mean | R | R<2K | R≥8K | draft ms | update ms | solve ms |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key in sorted(t, key=lambda k: (["baseline", "bias", "rls", "tts"].index(k[0]), k[1])):
            res = t[key]
            if "mean" not in res:
                continue
            mark = " **★**" if best.get(key[0]) == key[1] else ""
            r = [fmt(R(t, key, base, tts, m)) for m in ("tau", "tau_lt2k", "tau_ge8k")]
            lines.append(f"| {key[0]} | {key[1]}{mark} | {fmt(res['aime']['tau'])} | {fmt(res['lcb']['tau'])} | "
                         f"{fmt(res['mean']['tau'])} | {fmt(res['mean']['tau_lt2k'])} | {fmt(res['mean']['tau_ge8k'])} | "
                         f"{r[0]} | {r[1]} | {r[2]} | {res['mean']['draft_ms']:.2f} | {res['mean']['update_ms']:.2f} | "
                         f"{res['mean']['solve_ms']:.2f} |")
        lines.append("")
    r_best = R(tab, ("rls", best["rls"]), base, tts)
    lines += [f"**Selected (★):** {json.dumps(best)}; TTS context cache: {'refresh' if refresh_used else 'stale'}.", "",
              f"**Dev R (best RLS, truncated):** {r_best:.3f} → futility rule (§12.6, R < 0.15 ⇒ STOP): "
              f"{'STOP' if r_best < 0.15 else 'continue to Stage 4'}.", ""]
    open(os.path.join(ROOT, "results/dev_table.md"), "w").write("\n".join(lines))
    sig = json.load(open(os.path.join(ROOT, "configs/sigma2.json")))["sigma2"]
    frozen = []
    for m in ("baseline", "bias", "rls", "tts"):
        cfg = next(r["header"]["config_full"] for r in runs if r["header"]["method"] == m and r["header"]["config"] == best[m])
        frozen.append({"method": m, "config": cfg})
    yaml.safe_dump({"sigma2": sig, "tts_cache": "refresh" if refresh_used else "stale", "dev_R": r_best,
                    "plan": frozen}, open(os.path.join(ROOT, "configs/frozen.yaml"), "w"), sort_keys=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
