"""Real-model exactness check of the batched pooled update (Step 1): drafter + lm_head/embeddings cast to fp32 after a
bf16 teacher-forced pass, S2 + TTS stride 10 pooled batched with `check: true` (per-round accumulated gradient
computed alongside on the same trajectory). Separates implementation error from bf16 rounding.

  python scripts/batched_check_fp32.py --trace aime24_II_01 --new 2400 --rounds 450 [--dtype bf16]
"""
import argparse
import json

import torch

from rlsdk.adapters import TTS
from rlsdk.pool import load_models_split
from rlsdk.replay import simulate, teacher_forced
from rlsdk.run import load_traces
from rlsdk.target_attn import default_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="aime24_II_01")
    ap.add_argument("--new", type=int, default=2400)
    ap.add_argument("--rounds", type=int, default=450)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    target, drafter = load_models_split("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16", dev)
    tr = load_traces("traces/traces.jsonl", None, [a.trace])[0]
    P = tr["prompt_len"]
    ids = torch.tensor(tr["prompt_ids"] + tr["output_ids"][:a.new], dtype=torch.long, device=dev)
    target.model.layers.to(dev)
    tf = teacher_forced(target, ids, drafter.target_layer_ids, attn_spec=default_spec(target))
    target.model.layers.to("cpu")
    torch.cuda.empty_cache()
    if a.dtype == "fp32":
        drafter.float()
        target.model.embed_tokens.float()
        target.lm_head.float()
        tf.ctx, tf.hstar = tf.ctx.float(), tf.hstar.float()
    pristine = {k: v.detach().clone() for k, v in drafter.state_dict().items()}
    ad = TTS({"lr": 1e-5, "ctx": "s2", "stride": 10, "pool": True, "batched": True, "check": True})
    ad.bind(drafter, pristine)
    rounds = simulate(drafter, target, ids, P, tf, ad, a.new, max_rounds=a.rounds)
    errs = ad.check_err
    # steps whose 10 rounds all had far chunks active
    upd = [r for r in rounds if r["m"] > 0]
    far_steps = [errs[i] for i in range(len(errs)) if all(r.get("far", 0) > 0 for r in upd[10 * i:10 * i + 10])]
    res = {"dtype": a.dtype, "trace": a.trace, "rounds": len(rounds), "steps": len(errs),
           "max_rel_err": max(errs), "median_rel_err": sorted(errs)[len(errs) // 2],
           "steps_far_active": len(far_steps), "max_rel_err_far_active": max(far_steps) if far_steps else None}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
