"""Amendment 1 checks on one dev trace (env-dflash, GPU):
 (a) lambda = 1: recursion (prior=decay) vs fixed-prior solver -> relative error of M and of dM (< 1e-4 required on M)
 (b) fp32 vs fp64 fixed solver: max |corrected logits diff| (> 1e-3 -> switch RLS to fp64)
Both run in lockstep along the primary arm's trajectory (shadows see exactly the same z, h* stream).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.adapters import RLS  # noqa: E402
from rlsdk.replay import simulate, teacher_forced  # noqa: E402
from rlsdk.run import load_models, load_traces  # noqa: E402


class Lockstep:
    """Drafts with `primary`; feeds identical updates to `shadow`; records diagnostics."""

    method = "rls_check"

    def __init__(self, primary, shadow, head_w64, mode, every=100):
        self.p, self.s, self.W, self.mode, self.every = primary, shadow, head_w64, mode, every
        self.max_logit_diff = 0.0
        self.m_rel = []

    def reset(self, d, device):
        self.p.reset(d, device)
        self.s.reset(d, device)
        self.n = 0

    def _corr(self, ad, z):
        st = ad.groups[0][2]
        zz = z.to(st.dtype)
        return zz + st.correction(zz)

    def apply(self, z, slots):
        if self.mode == "fp64":
            a = self._corr(self.p, z.float()).double()
            b = self._corr(self.s, z.double())
            self.max_logit_diff = max(self.max_logit_diff, float(((a - b) @ self.W.T).abs().max()))
        return self.p.apply(z, slots)

    def update(self, z, h, slots, rnd):
        self.p.update(z, h, slots, rnd)
        self.s.update(z, h, slots, rnd)
        self.n += 1
        if self.mode == "lam1" and self.n % self.every == 0:
            self.m_rel.append(self._mrel())

    def _mrel(self):
        A = self.p.groups[0][2].M(torch.float64)
        B = self.s.groups[0][2].M(torch.float64)
        I = torch.eye(A.shape[0], A.shape[1], dtype=torch.float64)
        return {"round": self.n, "rel_M": float((A - B).norm() / B.norm()),
                "rel_dM": float((A - B).norm() / max(float((B - I).norm()), 1e-30))}

    def pop_solve_ms(self):
        return self.p.pop_solve_ms() + self.s.pop_solve_ms()

    def state_bytes(self):
        return self.p.state_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--trace-id", required=True)
    ap.add_argument("--sigma2", type=float, required=True)
    ap.add_argument("--n0", type=float, default=16)
    ap.add_argument("--lam-fp64", type=float, default=0.995)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    args.tiny = False
    args.model, args.draft = "Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"
    target, drafter = load_models(args, device)
    tr = load_traces(args.traces, ids=[args.trace_id])[0]
    ids = torch.tensor(tr["prompt_ids"] + tr["output_ids"], device=device)
    tf = teacher_forced(target, ids, drafter.target_layer_ids)
    W64 = target.lm_head.weight.double()
    base = {"n0": args.n0, "sigma2": args.sigma2}
    out = {"trace_id": args.trace_id, "n0": args.n0, "sigma2": args.sigma2}

    lk = Lockstep(RLS({**base, "lam": 1.0, "prior": "decay"}), RLS({**base, "lam": 1.0, "prior": "fixed"}), W64, "lam1")
    rounds = simulate(drafter, target, ids, tr["prompt_len"], tf, lk, args.max_new_tokens)
    lk.m_rel.append(lk._mrel())
    out["lam1"] = {"rounds": len(rounds), "trajectory": lk.m_rel, "final": lk.m_rel[-1],
                   "pass": lk.m_rel[-1]["rel_M"] < 1e-4}
    print(json.dumps(out["lam1"]["final"]), flush=True)

    lk = Lockstep(RLS({**base, "lam": args.lam_fp64, "prior": "fixed", "dtype": "fp32"}),
                  RLS({**base, "lam": args.lam_fp64, "prior": "fixed", "dtype": "fp64"}), W64, "fp64")
    rounds = simulate(drafter, target, ids, tr["prompt_len"], tf, lk, args.max_new_tokens)
    out["fp64"] = {"lam": args.lam_fp64, "rounds": len(rounds), "max_logit_diff": lk.max_logit_diff,
                   "fp32_cholesky_fallback": lk.p.diagnostics()["fp64_fallback_at"],
                   "switch_to_fp64": lk.max_logit_diff > 1e-3 or lk.p.diagnostics()["fp64_fallback_at"][0] is not None}
    print(json.dumps(out["fp64"]), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
