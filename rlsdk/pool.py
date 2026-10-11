"""Concurrent replay worker (Amendment §12.3). Several of these run side by side on one GPU under MPS.

Job = (trace, method, config). A worker claims a job with an O_EXCL claim file holding its pid; claims of dead pids
are reclaimed, so a killed pool resumes cleanly. Outputs are the same per-(trace, method, config) files as
rlsdk.run (temp file + rename), so finished jobs are skipped.

GPU memory: the target's 36 decoder layers live on CPU; only embeddings/lm_head/norm stay on GPU (the replay needs
them). For the teacher-forced pass a worker takes a file lock, moves the layers to GPU, runs the pass, and moves
them back -- so at most one full target is on the GPU at a time.
"""
import argparse
import os as _os

_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (TTS updates)
import fcntl
import gc
import json
import os
import socket
import time

import torch
import yaml

from .replay import near_tie_rate, teacher_forced
from .run import attn_spec, check_near_tie, config_name, load_traces, out_path, run_one


def load_models_split(model_id, draft_id, device, tiny=False):
    if tiny:  # dry runs: random tiny models, same code path
        from .tiny import tiny_drafter, tiny_target
        target = tiny_target()
        drafter = tiny_drafter(target).to(device)
        for p in target.parameters():
            p.requires_grad_(False)
        target.model.embed_tokens.to(device)
        target.model.norm.to(device)
        target.model.rotary_emb.to(device)
        target.lm_head.to(device)
        return target, drafter
    from dflash.model import DFlashDraftModel
    from transformers import AutoModelForCausalLM
    target = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", dtype=torch.bfloat16).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    target.model.embed_tokens.to(device)
    target.model.norm.to(device)
    target.model.rotary_emb.to(device)
    target.lm_head.to(device)  # tied to embed_tokens
    drafter = DFlashDraftModel.from_pretrained(draft_id, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    return target, drafter


def tf_pass_locked(target, drafter, ids, lock_path, device, spec=None):
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            target.model.layers.to(device)
            tf = teacher_forced(target, ids, drafter.target_layer_ids, attn_spec=spec)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            target.model.layers.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            fcntl.flock(lf, fcntl.LOCK_UN)
    return tf


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def try_claim(path):
    claim = path + ".claim"
    me = {"host": socket.gethostname(), "pid": os.getpid()}
    for _ in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps(me).encode())
            os.close(fd)
            return claim
        except FileExistsError:
            try:
                owner = json.load(open(claim))
            except (OSError, ValueError):
                return None
            if owner.get("host") == me["host"] and not _alive(owner["pid"]):
                os.remove(claim)  # stale claim from a dead worker
                continue
            return None
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--kind", choices=["heavy", "light", "any"], default="any", help="heavy = TTS jobs")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--near-tie-max", type=float, default=0.03)
    ap.add_argument("--tf-lock", default=None, help="shared lock file across workers (default: OUT/.tf.lock)")
    ap.add_argument("--worker-index", type=int, default=0, help="staggers start traces across workers")
    ap.add_argument("--tiny", action="store_true", help="dry run with the random tiny models")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    plan = yaml.safe_load(open(args.plan))
    traces = load_traces(args.traces, args.split, args.ids)
    jobs = [(tr, p["method"], p.get("config", {})) for tr in traces for p in plan
            if args.kind == "any" or (p["method"] == "tts") == (args.kind == "heavy")]
    lock = args.tf_lock or os.path.join(args.out, ".tf.lock")
    target = drafter = pristine = None
    cached = (None, None, None)  # (trace_id, tf, near_tie)
    order = {tr["trace_id"]: i for i, tr in enumerate(traces)}
    n_tr = max(len(traces), 1)
    skip = set()  # jobs claimed by someone else this pass
    while True:
        # prefer jobs on the trace whose teacher-forced pass is cached; otherwise start from a staggered trace
        todo = [j for j in jobs if not os.path.exists(out_path(args.out, j[0]["trace_id"], j[1], j[2]))
                and id(j) not in skip]
        if not todo:
            break
        todo.sort(key=lambda j: (j[0]["trace_id"] != cached[0],
                                 (order[j[0]["trace_id"]] - args.worker_index * n_tr // 5) % n_tr))
        job = todo[0]
        tr, method, cfg = job
        path = out_path(args.out, tr["trace_id"], method, cfg)
        claim = try_claim(path)
        if claim is None:
            skip.add(id(job))
            continue
        try:
            if os.path.exists(path):
                continue
            if target is None:
                target, drafter = load_models_split(args.model, args.draft, device, args.tiny)
                pristine = {k: v.detach().clone() for k, v in drafter.state_dict().items()}
            ids = torch.tensor(tr["prompt_ids"] + tr["output_ids"], dtype=torch.long, device=device)
            if cached[0] != tr["trace_id"]:
                cached = (None, None, None)
                gc.collect()  # free the previous jobs' adapter state (reference cycles) before the TF pass
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                tf = tf_pass_locked(target, drafter, ids, lock, device, attn_spec(plan, target))
                nt = near_tie_rate(tf.argmax, ids, tr["prompt_len"])
                cached = (tr["trace_id"], tf, nt)
                print(f"[{os.getpid()}] [{tr['trace_id']}] tf_pass={time.time() - t0:.1f}s near_tie={nt:.4%}", flush=True)
                check_near_tie(tr["trace_id"], nt, args.near_tie_max)
            _, tf, nt = cached
            s = run_one(tr, method, cfg, target, drafter, pristine, tf, ids, nt, args, path)
            peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
            print(f"[{os.getpid()}] [{tr['trace_id']}] {method} {config_name(cfg)} tau={s['tau_mean']:.3f} "
                  f"rounds={s['n_rounds']} wall={s['wall_s']:.1f}s peak_reserved={peak:.2f}GB", flush=True)
        finally:
            if os.path.exists(claim):
                os.remove(claim)
    print(f"[{os.getpid()}] WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
