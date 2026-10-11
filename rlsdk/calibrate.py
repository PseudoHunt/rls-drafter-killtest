"""Session 3, Step 3: one-time offline calibration of the DFlash drafter on thinking-on calibration traces, in S2 layout.

Sample = anchor t (uniform over output positions with t + 15 <= last position, pooled over traces; seed 0). Drafter
input exactly as the S2 replay drafts at frontier t: visible context = prompt + last W = 1024 positions + 16 far
32-token chunks scored by the target's attention row t-1 (layers 17/25/33), chosen by the same code as the replay
(`WindowCache.begin`), at contiguous positions 0..L-1, block [s_t, MASK x 15] at L..L+15. With contiguous positions this
is the stock DFlash forward on the visible rows (no cache), so the context K/V are built in-graph with the current
weights (teacher-forced). Loss = sum_i w_i KL(softmax(lm_head h*_{t+i-1}) || softmax(lm_head z_i)), i = 1..15,
w_i = exp(-(i-1)/7); mean over the batch. Adam (0.9, 0.999, eps 1e-8, constant lr) on fp32 masters, bf16 drafting copy.

Teacher-forced features are computed per trace once per run and held in CPU RAM; nothing is written to disk except
the final checkpoint (save_pretrained, bf16) and a JSON log.

  python -m rlsdk.calibrate --traces traces/traces_cal.jsonl --lrs 3e-6 1e-5 --steps N --batch 16 --out ckpt/cal
  python -m rlsdk.calibrate ... --probe 20      # timing only: N steps, no checkpoint, no evaluation
"""
import argparse
import hashlib
import json
import os
import random
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from dflash.model import _output_head, _raw_input_embeddings, _draft_value

from .adapters import dflash_position_weights
from .data import loop_cut
from .replay import teacher_forced
from .target_attn import default_spec
from .window import WindowCache

W, N_FAR, CHUNK = 1024, 16, 32


def load_cal_traces(path):
    """Calibration traces, loop-flagged ones truncated at the loop cut (§12.4 rule); dropped ones removed."""
    out = []
    for line in open(path):
        tr = json.loads(line)
        ids = tr["output_ids"]
        if tr.get("loop"):
            cut = loop_cut(ids)
            if cut["dropped"]:
                continue
            ids = ids[:cut["c"]]
        out.append({**tr, "output_ids": ids})
    return out


def featurize(target, drafter, traces, device, log=print):
    """Teacher-forced pass per trace -> CPU tensors (ctx features, h*, S2 score rows)."""
    spec = {"mid3": default_spec(target)["mid3"]}
    feats = []
    target.model.layers.to(device)
    for tr in traces:
        t0 = time.time()
        ids = torch.tensor(tr["prompt_ids"] + tr["output_ids"], dtype=torch.long, device=device)
        tf = teacher_forced(target, ids, drafter.target_layer_ids, attn_spec=spec)
        feats.append({"trace_id": tr["trace_id"], "P": tr["prompt_len"], "ids": ids.cpu(),
                      "ctx": tf.ctx.cpu(), "hstar": tf.hstar.cpu(), "score": tf.attn["mid3"].float().cpu()})
        del tf
        log(f"tf {tr['trace_id']} T={ids.shape[0]} {time.time() - t0:.1f}s")
    target.model.layers.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats


def anchors(feats, n, seed=0):
    """n anchors (trace index, t), uniform over all valid output positions pooled over traces."""
    spans = [(i, f["P"], f["ids"].shape[0] - 16) for i, f in enumerate(feats)]  # t in [P, T-16]
    sizes = [max(0, hi - lo + 1) for _, lo, hi in spans]
    cum, total = [], 0
    for s in sizes:
        total += s
        cum.append(total)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = rng.randrange(total)
        i = next(k for k, c in enumerate(cum) if u < c)
        out.append((i, spans[i][1] + u - (cum[i] - sizes[i])))
    return out


def visible_rows(drafter, f, t, device, layout="s2"):
    """S2 visible context positions at frontier t (the replay's own selection code); layout "full" (cal-full,
    session 4): every position [0, t) at its own position, i.e. the stock full-context replay's drafter input."""
    if layout == "full":
        return torch.arange(t)
    wc = WindowCache(drafter, f["P"], W, 1, device, next(drafter.parameters()).dtype,
                     far={"n": N_FAR, "chunk": CHUNK, "score": f["score"], "seed": 0})
    wc.begin(0, t, 16)
    return wc.visible_positions().cpu()


def build_batch(drafter, target, feats, batch, device, layout="s2"):
    """Padded S2-layout (or full-layout) batch: ctx rows [B, Lmax, F], block embeddings, position ids, bool mask, targets."""
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    vis = [visible_rows(drafter, feats[i], t, device, layout) for i, t in batch]
    Ls = [v.numel() for v in vis]
    Lm = max(Ls)
    F = feats[0]["ctx"].shape[-1]
    dt = next(drafter.parameters()).dtype
    th = torch.zeros((len(batch), Lm, F), dtype=dt)
    pos = torch.zeros((len(batch), Lm + 16), dtype=torch.long)
    mask = torch.zeros((len(batch), 1, 16, Lm + 16), dtype=torch.bool)
    blocks = torch.full((len(batch), 16), drafter.mask_token_id, dtype=torch.long)
    hs = []
    for b, ((i, t), v, L) in enumerate(zip(batch, vis, Ls)):
        f = feats[i]
        th[b, :L] = f["ctx"][v]
        pos[b, :L] = torch.arange(L)
        pos[b, Lm:] = torch.arange(L, L + 16)
        mask[b, :, :, :L] = True
        mask[b, :, :, Lm:] = True
        blocks[b, 0] = f["ids"][t]
        hs.append(f["hstar"][t:t + 15])
    noise = _raw_input_embeddings(target, blocks.to(device), scale)
    return th.to(device), noise, pos.to(device), mask.to(device), torch.stack(hs).to(device)


def s2_forward(drafter, th, noise, pos, mask):
    """Stock DFlash drafter forward (no cache) on the visible rows; returns z for slots 1..15, [B, 15, H]."""
    with sdpa_kernel(SDPBackend.MATH):
        return drafter(target_hidden=th, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                       past_key_values=None, use_cache=False)[:, 1 - 16:, :]


def batch_loss(drafter, head, z, hs, weights):
    B = z.shape[0]
    q_logp = torch.log_softmax(drafter.compute_logits(z.reshape(1, B * 15, -1), head)[0].float(), dim=-1)
    with torch.no_grad():
        p_logp = torch.log_softmax(head(hs.reshape(B * 15, -1)).float(), dim=-1)
    kl = (p_logp.exp() * (p_logp - q_logp)).sum(-1).view(B, 15)
    return (kl * weights[None, :15]).sum(1).mean()


def train(drafter, target, feats, lr, steps, batch_size, device, seed=0, log=print, pristine=None, layout="s2",
          micro=1):
    """Adam on fp32 masters; returns per-step losses. Drafter ends with the trained bf16 weights.
    micro > 1: the step's anchors run as `micro` consecutive sub-batches (forward + backward each, loss weighted by
    its share of the batch, gradients summed in fp32 on the masters) = the same mean-over-batch gradient."""
    head = _output_head(target)
    if pristine is not None:
        drafter.load_state_dict(pristine)
    params = list(drafter.parameters())
    for p in params:
        p.requires_grad_(True)
    masters = [p.detach().float().clone() for p in params]
    opt = torch.optim.Adam(masters, lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=device.type == "cuda")
    weights = dflash_position_weights(16, 7.0).to(device)
    sched = anchors(feats, steps * batch_size, seed)
    losses = []
    t0 = time.time()
    for s in range(steps):
        step_anchors = sched[s * batch_size:(s + 1) * batch_size]
        if micro > 1:  # same anchors, same mean gradient; sub-batches of similar length pad less
            step_anchors = sorted(step_anchors, key=lambda it: it[1])
        size = -(-batch_size // micro)
        loss = 0.0
        for mp in masters:
            mp.grad = None
        for j in range(0, batch_size, size):
            part = step_anchors[j:j + size]
            th, noise, pos, mask, hs = build_batch(drafter, target, feats, part, device, layout)
            with torch.enable_grad():
                lp = batch_loss(drafter, head, s2_forward(drafter, th, noise, pos, mask), hs, weights)
                (lp * (len(part) / batch_size) if micro > 1 else lp).backward()
            del th, noise, pos, mask, hs
            loss = loss + lp.detach() * (len(part) / batch_size)
            for p, mp in zip(params, masters):
                if p.grad is not None:
                    mp.grad = p.grad.float() if mp.grad is None else mp.grad + p.grad.float()
                p.grad = None
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, mp in zip(params, masters):
                p.copy_(mp)
        losses.append(float(loss.detach()))
        if s % 50 == 0 or s == steps - 1:
            log(f"lr={lr:g} step {s + 1}/{steps} loss={losses[-1]:.4f} "
                f"mean50={sum(losses[-50:]) / len(losses[-50:]):.4f} {time.time() - t0:.0f}s")
    for p in params:
        p.requires_grad_(False)
    return losses


def sha256_dir(path):
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            with open(os.path.join(path, name), "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-6, 1e-5])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="ckpt/cal")
    ap.add_argument("--probe", type=int, default=None, help="timing only: run this many steps, save nothing")
    ap.add_argument("--budget-h", type=float, default=None,
                    help="fix --steps from a timing probe (20 steps, discarded) so the whole run fits this budget")
    ap.add_argument("--margin-s", type=float, default=240.0, help="reserved for checkpoint saves and slack")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="first N traces only (dry runs)")
    ap.add_argument("--layout", choices=["s2", "full"], default="s2", help="full = cal-full (session 4)")
    ap.add_argument("--micro", type=int, default=1, help="sub-batches per step (memory; same gradient)")
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(a.device)
    t_start = time.time()
    from .pool import load_models_split
    target, drafter = load_models_split(a.model, a.draft, device, a.tiny)
    pristine = {k: v.detach().clone() for k, v in drafter.state_dict().items()}
    traces = load_cal_traces(a.traces)[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.txt" if a.probe is None else "probe_log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"{len(traces)} calibration traces, {sum(len(t['output_ids']) for t in traces)} output tokens")
    t0 = time.time()
    feats = featurize(target, drafter, traces, device, log)
    log(f"teacher-forced passes: {time.time() - t0:.0f}s")
    if a.probe is not None:
        t1 = time.time()
        train(drafter, target, feats, a.lrs[0], a.probe, a.batch, device, log=log, pristine=pristine, layout=a.layout, micro=a.micro)
        if device.type == "cuda":
            torch.cuda.synchronize()
        log(f"PROBE {a.probe} steps x batch {a.batch}: {(time.time() - t1) / a.probe:.3f} s/step; "
            f"peak {torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0:.1f} GiB")
        return
    if a.budget_h is not None:  # timing-only probe; its weights are discarded (train() restarts from pristine)
        n_probe = 20
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        train(drafter, target, feats, a.lrs[0], n_probe, a.batch, device, log=lambda m: None, pristine=pristine,
              layout=a.layout, micro=a.micro)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sps = (time.time() - t1) / n_probe
        left = a.budget_h * 3600 - (time.time() - t_start) - a.margin_s
        a.steps = max(1, int(left / (len(a.lrs) * sps)))
        log(f"STEPS FIXED: probe {sps:.3f} s/step (batch {a.batch}); {left:.0f}s left for {len(a.lrs)} lrs -> "
            f"steps = {a.steps} ({a.steps * a.batch} anchors per lr)")
    summary = {"traces": [t["trace_id"] for t in traces], "steps": a.steps, "batch": a.batch, "layout": a.layout,
               "micro": a.micro, "runs": {}}
    for lr in a.lrs:
        t1 = time.time()
        losses = train(drafter, target, feats, lr, a.steps, a.batch, device, log=log, pristine=pristine,
                       layout=a.layout, micro=a.micro)
        d = os.path.join(a.out, f"lr{lr:g}")
        drafter.save_pretrained(d)
        sha = sha256_dir(d)
        summary["runs"][f"{lr:g}"] = {"dir": d, "sha256": sha, "train_s": time.time() - t1,
                                      "loss_first50": sum(losses[:50]) / min(50, len(losses)),
                                      "loss_last50": sum(losses[-50:]) / min(50, len(losses)), "losses": losses}
        log(f"saved {d} sha256={sha} train {time.time() - t1:.0f}s")
    drafter.load_state_dict(pristine)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    log("CALIBRATE_DONE")


if __name__ == "__main__":
    main()
