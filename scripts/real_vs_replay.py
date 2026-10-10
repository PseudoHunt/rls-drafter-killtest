"""Stage 2 (env-dflash): actual DFlash greedy speculative generate on 3 prompts (2 AIME + 1 LCB, dev
split), per-round acceptance logged via return_stats, then replay of the same outputs.
Pass: |mean tau diff| <= 1% and >= 98% of real rounds identical."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dflash.benchmark import load_transformers_models, stop_token_ids  # noqa: E402
from dflash.model import dflash_generate  # noqa: E402

from rlsdk.adapters import Baseline  # noqa: E402
from rlsdk.replay import near_tie_rate, simulate, teacher_forced  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="traces.jsonl (for prompt ids / split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    traces = [json.loads(line) for line in open(args.traces)]
    dev = [t for t in traces if t["split"] == "dev"]
    pick = [t for t in dev if t["dataset"] == "aime"][:2] + [t for t in dev if t["dataset"] == "lcb"][:1]

    target, drafter, tok = load_transformers_models("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16", device)
    stops = stop_token_ids(target, tok)
    results = []
    for tr in pick:
        prompt = torch.tensor([tr["prompt_ids"]], device=device)
        t0 = time.time()
        out = dflash_generate(drafter, target, prompt, args.max_new_tokens, stops, return_stats=True)
        gen_s = time.time() - t0
        ids = out.output_ids[0]
        P = tr["prompt_len"]
        with torch.no_grad():
            tf = teacher_forced(target, ids, drafter.target_layer_ids)
        nt = near_tie_rate(tf.argmax, ids, P)
        rounds = simulate(drafter, target, ids, P, tf, Baseline(), args.max_new_tokens)
        real = out.acceptance_lengths
        rep = [r["tau"] for r in rounds]
        # real round start positions
        real_t, t = [], P
        for a in real:
            real_t.append(t)
            t += a
        rep_by_t = {r["t"]: r["tau"] for r in rounds}
        same_idx = sum(1 for i, a in enumerate(real) if i < len(rep) and rep[i] == a and rounds[i]["t"] == real_t[i])
        same_t = sum(1 for tt, a in zip(real_t, real) if rep_by_t.get(tt) == a)
        res = {"trace_id": tr["trace_id"], "n_out": int(ids.shape[0] - P), "real_rounds": len(real),
               "replay_rounds": len(rep), "tau_real": sum(real) / len(real), "tau_replay": sum(rep) / len(rep),
               "frac_identical_index": same_idx / len(real), "frac_identical_by_t": same_t / len(real),
               "near_tie_rate": nt, "gen_s": round(gen_s, 1),
               "vllm_prefix_match": int(next((i for i, (a, b) in enumerate(zip(ids[P:].tolist(), tr["output_ids"])) if a != b),
                                             min(ids.shape[0] - P, len(tr["output_ids"]))))}
        print(json.dumps(res), flush=True)
        results.append({**res, "real": real, "replay": rep})
    tot_real = [a for r in results for a in r["real"]]
    tot_rep = [a for r in results for a in r["replay"]]
    n_same = sum(r["frac_identical_index"] * r["real_rounds"] for r in results)
    agg = {"tau_real": sum(tot_real) / len(tot_real), "tau_replay": sum(tot_rep) / len(tot_rep),
           "frac_identical_index": n_same / len(tot_real)}
    agg["tau_rel_diff"] = abs(agg["tau_replay"] - agg["tau_real"]) / agg["tau_real"]
    agg["pass"] = agg["tau_rel_diff"] <= 0.01 and agg["frac_identical_index"] >= 0.98
    print(json.dumps(agg), flush=True)
    json.dump({"per_trace": results, "aggregate": agg}, open(args.out, "w"))


if __name__ == "__main__":
    main()
