"""Resumable replay runner: one output file per (trace, method, config); existing files are skipped.

Output: results/raw/{trace_id}__{method}__{config}.jsonl  (header line, one line per round, summary).
Files are written to a .tmp name and renamed when complete.
"""
import argparse
import json
import os
import time

import torch
import yaml

from .adapters import RLS, TTS, Baseline, Bias
from .replay import near_tie_rate, simulate, teacher_forced

METHODS = {"baseline": Baseline, "bias": Bias, "rls": RLS, "rls_slot": RLS, "tts": TTS}


def config_name(cfg):
    keys = sorted(k for k in cfg if k != "sigma2")
    return "_".join(f"{k}={cfg[k]}" for k in keys) or "default"


def out_path(out_dir, trace_id, method, cfg):
    return os.path.join(out_dir, f"{trace_id}__{method}__{config_name(cfg)}.jsonl")


def load_traces(path, split=None, ids=None):
    traces = [json.loads(line) for line in open(path)]
    if split not in (None, "all"):
        traces = [t for t in traces if t["split"] == split]
    if ids:
        traces = [t for t in traces if t["trace_id"] in ids]
    return traces


def make_adapter(method, cfg, drafter, pristine):
    cfg = dict(cfg)
    if method == "rls_slot":
        cfg["per_slot"] = True
    ad = METHODS[method](cfg) if method != "baseline" else Baseline(cfg)
    if method == "tts":
        ad.bind(drafter, pristine)
    return ad


def load_models(args, device):
    if args.tiny:
        from .tiny import tiny_drafter, tiny_target
        target = tiny_target().to(device)
        drafter = tiny_drafter(target).to(device)
        return target, drafter
    from dflash.benchmark import load_transformers_models
    target, drafter, _ = load_transformers_models(args.model, args.draft, device)
    for p in list(target.parameters()) + list(drafter.parameters()):
        p.requires_grad_(False)
    return target, drafter


def run_one(trace, method, cfg, target, drafter, pristine, tf, ids, nt, args, path):
    ad = make_adapter(method, cfg, drafter, pristine)
    sleep = float(os.environ.get("RLSDK_SLEEP_PER_ROUND", "0"))
    rows = []

    def on_round(row):
        rows.append(row)
        if sleep:
            time.sleep(sleep)

    t0 = time.time()
    rounds = simulate(drafter, target, ids, trace["prompt_len"], tf, ad, args.max_new_tokens, on_round=on_round)
    state_bytes = ad.state_bytes()
    if method == "tts":
        ad.finish()
    head = {"type": "header", "trace_id": trace["trace_id"], "dataset": trace["dataset"], "split": trace["split"],
            "method": method, "config": config_name(cfg), "config_full": cfg, "prompt_len": trace["prompt_len"],
            "n_out": len(trace["output_ids"]), "finish_reason": trace.get("finish_reason"),
            "loop": trace.get("loop", False), "near_tie_rate": nt, "state_bytes": state_bytes}
    summ = {"type": "summary", "n_rounds": len(rounds), "tau_mean": sum(r["tau"] for r in rounds) / max(len(rounds), 1),
            "wall_s": round(time.time() - t0, 3)}
    if method == "baseline":
        summ.update(z2_sum=ad.z2_sum, z2_count=ad.z2_count)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(head) + "\n")
        for r in rounds:
            f.write(json.dumps({"type": "round", "trace_id": trace["trace_id"], "method": method,
                                "config": config_name(cfg), **r}) + "\n")
        f.write(json.dumps(summ) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--plan", required=True, help="yaml: list of {method, config}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--near-tie-max", type=float, default=0.01)
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    plan = yaml.safe_load(open(args.plan))
    os.makedirs(args.out, exist_ok=True)
    traces = load_traces(args.traces, args.split, args.ids)

    target = drafter = pristine = None
    for trace in traces:
        todo = [(p["method"], p.get("config", {})) for p in plan
                if not os.path.exists(out_path(args.out, trace["trace_id"], p["method"], p.get("config", {})))]
        if not todo:
            continue
        if target is None:
            target, drafter = load_models(args, device)
            pristine = {k: v.detach().clone() for k, v in drafter.state_dict().items()}
        ids = torch.tensor(trace["prompt_ids"] + trace["output_ids"], dtype=torch.long, device=device)
        t0 = time.time()
        tf = teacher_forced(target, ids, drafter.target_layer_ids)
        nt = near_tie_rate(tf.argmax, ids, trace["prompt_len"])
        print(f"[{trace['trace_id']}] T={ids.shape[0]} tf_pass={time.time() - t0:.1f}s near_tie={nt:.4%}", flush=True)
        if nt > args.near_tie_max:
            print(f"[{trace['trace_id']}] WARNING near-tie rate {nt:.4%} above {args.near_tie_max:.2%}", flush=True)
        for method, cfg in todo:
            path = out_path(args.out, trace["trace_id"], method, cfg)
            s = run_one(trace, method, cfg, target, drafter, pristine, tf, ids, nt, args, path)
            print(f"[{trace['trace_id']}] {method} {config_name(cfg)} tau={s['tau_mean']:.3f} "
                  f"rounds={s['n_rounds']} wall={s['wall_s']:.1f}s", flush=True)
        del tf
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
