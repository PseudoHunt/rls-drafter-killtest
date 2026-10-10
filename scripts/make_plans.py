"""Write Stage-3 plan files. Grids fixed by the brief §5 + Amendment 1 (KILL_TEST_BRIEF.md §11).

  python scripts/make_plans.py baseline               -> configs/plan_dev_baseline.yaml
  python scripts/make_plans.py grid --sigma2 X        -> configs/plan_dev_grid.yaml (bias + RLS + TTS stale)
  python scripts/make_plans.py refresh --lr L [...]   -> configs/plan_dev_refresh.yaml
"""
import argparse
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIAS_RHO = [0.9, 0.99, 0.999]
RLS_N0 = [16, 128, 1024]
RLS_LAM = [0.995, 0.999, 0.9999, 1.0]
TTS_LR = [1e-5, 3e-5, 1e-4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["baseline", "grid", "refresh"])
    ap.add_argument("--sigma2", type=float)
    ap.add_argument("--lr", type=float, nargs="*")
    args = ap.parse_args()
    if args.what == "baseline":
        plan, name = [{"method": "baseline"}], "plan_dev_baseline.yaml"
    elif args.what == "grid":
        plan = [{"method": "bias", "config": {"rho": r}} for r in BIAS_RHO]
        plan += [{"method": "rls", "config": {"n0": n0, "lam": lam, "sigma2": args.sigma2}}
                 for n0 in RLS_N0 for lam in RLS_LAM]
        plan += [{"method": "tts", "config": {"lr": lr}} for lr in TTS_LR]
        name = "plan_dev_grid.yaml"
    else:
        plan = [{"method": "tts", "config": {"lr": lr, "refresh": True}} for lr in args.lr]
        name = "plan_dev_refresh.yaml"
    path = os.path.join(ROOT, "configs", name)
    yaml.safe_dump(plan, open(path, "w"), sort_keys=False)
    print(path, len(plan))


if __name__ == "__main__":
    main()
