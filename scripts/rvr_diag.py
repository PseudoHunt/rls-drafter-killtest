"""Stage 2 diagnostic: where do real-vs-replay round mismatches come from?

Runs real DFlash generate while recording the exact context features it feeds the drafter (monkeypatched
extract_context_feature -- wrapper, no source edit). Then replays the same output three ways:
  A. teacher-forced features (the harness default)
  B. the real run's recorded features (isolates harness logic from feature numerics)
and reports per-round agreement with the real run, plus feature differences between A and the real run.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dflash.model as dm  # noqa: E402
from dflash.benchmark import load_transformers_models, stop_token_ids  # noqa: E402

from rlsdk.adapters import Baseline  # noqa: E402
from rlsdk.replay import simulate, teacher_forced  # noqa: E402


def agreement(real, rounds, P):
    real_t, t = [], P
    for a in real:
        real_t.append(t)
        t += a
    by_t = {r["t"]: r["tau"] for r in rounds}
    same = sum(1 for tt, a in zip(real_t, real) if by_t.get(tt) == a)
    rep = [r["tau"] for r in rounds]
    return {"frac_identical_by_t": same / len(real), "tau_real": sum(real) / len(real), "tau_replay": sum(rep) / len(rep),
            "first_mismatch_round": next((i for i, (tt, a) in enumerate(zip(real_t, real)) if by_t.get(tt) != a), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--trace-id", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    tr = next(json.loads(line) for line in open(args.traces) if json.loads(line)["trace_id"] == args.trace_id)
    target, drafter, tok = load_transformers_models("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16", device)

    recorded = []
    orig = dm.extract_context_feature

    def rec(hs, ids):
        out = orig(hs, ids)
        recorded.append(out)
        return out

    dm.extract_context_feature = rec
    prompt = torch.tensor([tr["prompt_ids"]], device=device)
    out = dm.dflash_generate(drafter, target, prompt, args.max_new_tokens, stop_token_ids(target, tok), return_stats=True)
    dm.extract_context_feature = orig
    real = out.acceptance_lengths
    ids = out.output_ids[0]
    P = tr["prompt_len"]
    # real per-position features: prefill [0,P) then round r's first `produced` rows
    chunks = [recorded[0][0]] + [recorded[i + 1][0, :a] for i, a in enumerate(real) if i + 1 < len(recorded)]
    real_ctx = torch.cat(chunks)[: ids.shape[0]]

    tf = teacher_forced(target, ids, drafter.target_layer_ids)
    res = {"trace_id": args.trace_id}
    res["A_teacher_forced"] = agreement(real, simulate(drafter, target, ids, P, tf, Baseline(), args.max_new_tokens), P)
    n = min(real_ctx.shape[0], tf.ctx.shape[0])
    diff = (tf.ctx[:n].float() - real_ctx[:n].float()).abs()
    res["feature_diff"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                           "frac_elems_differ": float((diff > 0).float().mean()), "n_positions": n}
    tf.ctx = real_ctx
    res["B_recorded_real_features"] = agreement(real, simulate(drafter, target, ids, P, tf, Baseline(), args.max_new_tokens), P)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
