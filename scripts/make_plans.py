"""Write Stage-3 plan files. Grids fixed by the brief §5 + Amendment 1 (KILL_TEST_BRIEF.md §11).

  python scripts/make_plans.py baseline               -> configs/plan_dev_baseline.yaml
  python scripts/make_plans.py grid --sigma2 X        -> configs/plan_dev_grid.yaml (bias + RLS + TTS stale)
  python scripts/make_plans.py refresh --lr L [...]   -> configs/plan_dev_refresh.yaml
  python scripts/make_plans.py screen [--out PATH]    -> configs/plan_screen.yaml (baseline + KL-bias + KL-linear)
  python scripts/make_plans.py window [--out PATH]    -> configs/plan_window.yaml (baseline + windowed context)
  python scripts/make_plans.py mech [--out PATH]      -> configs/plan_mech.yaml (context policies B/E/C/C'/D)
"""
import argparse
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIAS_RHO = [0.9, 0.99, 0.999]
RLS_N0 = [16, 128, 1024]
RLS_LAM = [0.995, 0.999, 0.9999, 1.0]
TTS_LR = [1e-5, 3e-5, 1e-4]
KLBIAS_ALPHA = [0.005, 0.01, 0.025]  # rho = 0.9
KLLIN_LR = [1e-7, 1e-6, 1e-5]
WINDOW_W = [1024, 4096]
WINDOW_KEEP = [64, "prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["baseline", "grid", "refresh", "screen", "window", "mech"])
    ap.add_argument("--sigma2", type=float)
    ap.add_argument("--lr", type=float, nargs="*")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.what == "baseline":
        plan, name = [{"method": "baseline"}], "plan_dev_baseline.yaml"
    elif args.what == "grid":
        plan = [{"method": "bias", "config": {"rho": r}} for r in BIAS_RHO]
        plan += [{"method": "rls", "config": {"n0": n0, "lam": lam, "sigma2": args.sigma2}}
                 for n0 in RLS_N0 for lam in RLS_LAM]
        plan += [{"method": "tts", "config": {"lr": lr}} for lr in TTS_LR]
        name = "plan_dev_grid.yaml"
    elif args.what == "screen":
        plan = [{"method": "baseline"}]
        plan += [{"method": "klbias", "config": {"rho": 0.9, "alpha": a}} for a in KLBIAS_ALPHA]
        plan += [{"method": "kllinear", "config": {"lr": lr}} for lr in KLLIN_LR]
        name = "plan_screen.yaml"
    elif args.what == "window":
        plan = [{"method": "baseline"}]
        plan += [{"method": "window", "config": {"W": w, "keep": k}} for w in WINDOW_W for k in WINDOW_KEEP]
        name = "plan_window.yaml"
    elif args.what == "mech":  # NOTES.md §10; baseline and the frozen window already exist in results/raw_a30_window
        plan = [{"method": "baseline"}, {"method": "window", "config": {"W": 1024, "keep": "prompt"}},
                {"method": "winorig", "config": {"W": 1024, "keep": "prompt"}},
                {"method": "capctx", "config": {"cap": 4096}},
                {"method": "topk", "config": {"k": 1024}},
                {"method": "topkcap", "config": {"k": 1024, "cap": 4096}},
                {"method": "topkshared", "config": {"k": 1024, "cap": 4096}}]
        name = "plan_mech.yaml"
    else:
        plan = [{"method": "tts", "config": {"lr": lr, "refresh": True}} for lr in args.lr]
        name = "plan_dev_refresh.yaml"
    path = args.out or os.path.join(ROOT, "configs", name)
    yaml.safe_dump(plan, open(path, "w"), sort_keys=False)
    print(path, len(plan))


if __name__ == "__main__":
    main()
