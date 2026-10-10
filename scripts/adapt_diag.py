"""Diagnostic (CPU-capable): do bias/RLS reduce out-of-sample hidden-state error, and what do they do to draft
accuracy? Runs along the BASELINE trajectory of one trace; shadow adapters are updated with the valid pairs only
(exactly as in the method) and evaluated on all 15 slots of the NEXT round before seeing them (measurement only;
all-slot targets are read from the unguarded teacher-forced arrays for scoring, never for updates).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsdk.adapters import RLS, Baseline, Bias  # noqa: E402
from rlsdk.replay import simulate, teacher_forced  # noqa: E402


class Probe(Baseline):
    def __init__(self, shadows, head, tf, ids, P):
        super().__init__()
        self.sh, self.W, self.tf, self.ids, self.P = shadows, head, tf, ids, P
        self.stats = {name: {"l2": [], "acc": [], "slot_acc": torch.zeros(15), "n": 0} for name in ["base", *shadows]}
        self.t = None

    def reset(self, d, device):
        super().reset(d, device)
        for a in self.sh.values():
            a.reset(d, device)

    def score(self, t, z):
        S = z.shape[0]
        h = self.tf.hstar[t:t + S].float()
        truth = self.ids[t + 1:t + 1 + S]
        n = min(S, truth.shape[0], h.shape[0])
        if n < S:
            return
        for name, zz in [("base", z), *[(k, a.apply(z, torch.arange(1, S + 1))) for k, a in self.sh.items()]]:
            zf = zz.float()
            st = self.stats[name]
            st["l2"].append(float(((h - zf) ** 2).sum(-1).mean()))
            corr = (self.W(zz).argmax(-1) == truth).float()
            st["acc"].append(float(corr.mean()))
            st["slot_acc"] += corr.cpu()
            st["n"] += 1

    def apply(self, z, slots):
        self._z = z
        return super().apply(z, slots)

    def update(self, z, h, slots, rnd):
        self.score(rnd.t, self._z)  # out-of-sample: shadows have not seen this round yet
        for a in self.sh.values():
            a.update(z, h, slots, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--trace-id", required=True)
    ap.add_argument("--n-out", type=int, default=3000)
    ap.add_argument("--sigma2", type=float, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(16)
    from rlsdk.run import load_models
    args.tiny, args.model, args.draft = False, "Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"
    dev = torch.device(args.device)
    if dev.type == "cpu":
        from dflash.model import DFlashDraftModel
        from transformers import AutoModelForCausalLM
        target = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", dtype=torch.bfloat16).eval()
        drafter = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).eval()
    else:
        target, drafter = load_models(args, dev)
    tr = next(json.loads(line) for line in open(args.traces) if json.loads(line)["trace_id"] == args.trace_id)
    P = tr["prompt_len"]
    ids = torch.tensor(tr["prompt_ids"] + tr["output_ids"][: args.n_out], device=dev)
    with torch.no_grad():
        tf = teacher_forced(target, ids, drafter.target_layer_ids)
        head = target.lm_head
        hacc = float((head(tf.hstar[P - 1:-1]).argmax(-1) == ids[P:]).float().mean())
        sh = {"bias_rho0.99": Bias({"rho": 0.99}),
              "rls_n0=128_lam0.999": RLS({"n0": 128, "lam": 0.999, "sigma2": args.sigma2, "dtype": "fp64"}),
              "rls_n0=1024_lam1": RLS({"n0": 1024, "lam": 1.0, "sigma2": args.sigma2, "dtype": "fp64"})}
        probe = Probe(sh, head, tf, ids, P)
        rounds = simulate(drafter, target, ids, P, tf, probe, 16384)
    out = {"trace_id": args.trace_id, "n_out": args.n_out, "rounds": len(rounds),
           "target_hstar_argmax_acc": hacc,
           "mean_abs_hstar": float(tf.hstar[P:].float().abs().mean()), "rms_hstar": float(tf.hstar[P:].float().pow(2).mean().sqrt())}
    for name, st in probe.stats.items():
        k = len(st["l2"])
        third = max(k // 3, 1)
        out[name] = {"l2_all": sum(st["l2"]) / k, "acc_all": sum(st["acc"]) / k,
                     "l2_last_third": sum(st["l2"][-third:]) / third, "acc_last_third": sum(st["acc"][-third:]) / third,
                     "slot_acc": [round(float(x) / st["n"], 3) for x in st["slot_acc"]]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
