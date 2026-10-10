"""Stage 5: results/summary.md + plots (brief §8, amendments §11–§12)."""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.analyze import load_cuts, load_raw, plot_bins, table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
DS = ("aime", "lcb", "mean")


def f(x, n=3):
    return "–" if x is None or x != x else f"{x:.{n}f}"


def band(r):
    if r != r:
        return "n/a"
    if r >= 0.50:
        b = "GO"
    elif r >= 0.25:
        b = "GREY"
    else:
        b = "STOP"
    border = any(abs(r - th) <= 0.05 for th in (0.25, 0.50))
    return b + (" (borderline)" if border else "")


def r_of(tab, key, base, tts, ds="mean", metric="tau"):
    b, t = tab[base][ds][metric], tab[tts][ds][metric]
    return (tab[key][ds][metric] - b) / (t - b) if t != b else float("nan")


def main():
    frozen = yaml.safe_load(open(J("configs/frozen.yaml")))
    from rlsdk.run import config_name
    keys = {p["method"]: (p["method"], config_name(p["config"])) for p in frozen["plan"]}
    if os.path.exists(J("configs/frozen_slot.yaml")):
        p = yaml.safe_load(open(J("configs/frozen_slot.yaml")))
        keys["rls_slot"] = ("rls_slot", config_name(p["config"]))
    cuts = load_cuts(J("results/loop_cuts.csv"))
    runs = load_raw(J("results/raw"), "test")
    tabs = {v: table(runs, v, cuts) for v in ("trunc", "exclude", "loops")}
    iso = table(load_raw(J("results/timing_isolated")), "loops")
    base, tts, rls = keys["baseline"], keys["tts"], keys["rls"]
    L = ["# Kill test summary: closed-form (RLS) test-time drafter adaptation vs TTS", ""]

    t = tabs["trunc"]
    tau_b, tau_t = t[base]["mean"]["tau"], t[tts]["mean"]["tau"]
    R = r_of(t, rls, base, tts)
    if tau_t < 1.10 * tau_b:
        decision = f"**INVALID** — τ_TTS = {tau_t:.3f} < 1.10 × τ_base = {1.1 * tau_b:.3f} on test."
    else:
        decision = (f"**{band(R)}** — R = (τ_RLS − τ_base)/(τ_TTS − τ_base) = ({t[rls]['mean']['tau']:.3f} − {tau_b:.3f})"
                    f"/({tau_t:.3f} − {tau_b:.3f}) = **{R:.3f}** (test, mean of AIME and LCB, flagged traces truncated).")
    R_ex = r_of(tabs["exclude"], rls, base, tts)
    consistent = band(R).split()[0] == band(R_ex).split()[0]
    L += ["## Decision", "", decision, "",
          f"Robustness: exclusion variant R = {R_ex:.3f} ({band(R_ex)}); loops-included R = "
          f"{r_of(tabs['loops'], rls, base, tts):.3f} ({band(r_of(tabs['loops'], rls, base, tts))}).",
          "" if consistent else "**Truncation and exclusion land in different decision bands — GO is not called "
          "until this is discussed (§11 Amendment 2).**", ""]
    if "rls_slot" in keys:
        L += [f"GREY follow-up (§5.5 per-slot RLS): R = {r_of(t, keys['rls_slot'], base, tts):.3f} "
              f"({band(r_of(t, keys['rls_slot'], base, tts))}).", ""]

    for v, title in (("trunc", "Test results — flagged traces truncated at loop onset (decision)"),
                     ("exclude", "Test results — flagged traces excluded (robustness)"),
                     ("loops", "Test results — loops included (§3)")):
        tv = tabs[v]
        L += [f"## {title}", "",
              "| method | config | dataset | τ | τ <2K | τ ≥8K | R | R <2K | R ≥8K | rounds | traces |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, key in keys.items():
            for ds in DS:
                res = tv[key][ds]
                rr = [f(r_of(tv, key, base, tts, ds, mm)) for mm in ("tau", "tau_lt2k", "tau_ge8k")]
                extra = (res["n_rounds"], res["n_traces"]) if ds != "mean" else ("", "")
                L.append(f"| {m} | {key[1]} | {ds} | {f(res['tau'])} | {f(res['tau_lt2k'])} | {f(res['tau_ge8k'])} | "
                         f"{rr[0]} | {rr[1]} | {rr[2]} | {extra[0]} | {extra[1]} |")
        L.append("")

    L += ["## Cost (isolated runs: frozen configs, one process alone on the GPU, 2 dev traces)", "",
          "| method | update ms/round | solve ms/round | draft forward ms/round | adaptation state (MB) |", "|---|---|---|---|---|"]
    for m, key in keys.items():
        if key in iso:
            r = iso[key]
            ds = "mean" if "mean" in r else next(iter(r))
            L.append(f"| {m} | {f(r[ds]['update_ms'], 2)} | {f(r[ds]['solve_ms'], 2)} | {f(r[ds]['draft_ms'], 2)} | "
                     f"{f(r[ds]['state_mb'], 1)} |")
    L.append("")

    traces = [json.loads(line) for line in open(J("traces/traces.jsonl"))]
    flagged = [x for x in traces if x["loop"]]
    dropped = [k for k, c in cuts.items() if c is None]
    L += ["## Looping traces", "",
          f"{len(flagged)}/62 flagged (test: {sum(x['split'] == 'test' for x in flagged)}, dev: "
          f"{sum(x['split'] == 'dev' for x in flagged)}); truncated at loop onset per `results/loop_cuts.csv`; "
          f"dropped (<1024 tokens kept): {', '.join(dropped) or 'none'}. Results with loops included are in the table above.", ""]

    nt = load_raw(J("results/raw_nothink"))
    ncuts = load_cuts(J("results/loop_cuts_nothink.csv"))
    L += ["## Thinking-OFF control (6 dev AIME prompts, baseline only; not part of the decision)", "",
          "| variant | τ overall | τ <2K | rounds |", "|---|---|---|---|"]
    for v in ("trunc", "loops"):
        tv = table(nt, v, ncuts)
        r = tv[("baseline", "default")]["aime"]
        L.append(f"| {'truncated' if v == 'trunc' else 'loops included'} | {f(r['tau'])} | {f(r['tau_lt2k'])} | {r['n_rounds']} |")
    dev_on = table(load_raw(J("results/raw"), "dev"), "trunc", cuts)[base]["aime"]
    L += ["", f"Thinking-ON baseline on the same 6 dev AIME prompts (truncated): τ = {f(dev_on['tau'])}, "
          f"τ <2K = {f(dev_on['tau_lt2k'])}.", ""]

    L += ["## Deviations from the brief", "", open(J("results/deviations.md")).read() if os.path.exists(J("results/deviations.md")) else "", ""]
    open(J("results/summary.md"), "w").write("\n".join(L))
    os.makedirs(J("results/plots"), exist_ok=True)
    labels = list(keys)
    for ds in ("aime", "lcb"):
        plot_bins(tabs["trunc"], list(keys.values()), labels, ds, J(f"results/plots/tau_vs_position_{ds}.png"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
