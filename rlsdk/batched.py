"""Exact batched pooled TTS update (stride k, pool) for a WindowCache drafter context (window / S2).

Parameters change only at Adam steps, so the k rounds pooled into one step share the same parameters. Instead of a
backward every round, each round stores its exact loss inputs and the step runs one batched forward + backward:
  - the round's new context rows (target features, the drafter input whose K/V are built in the graph, with grad),
  - the block's input embeddings and the queries' positions,
  - the indices of the older visible context rows in the WindowCache buffers. The buffers are append-only (a
    position is written once, when it is first fed, with the parameters of that time), so indexing them at step
    time returns exactly the stale K/V the round attended to -- never K/V rebuilt with the current weights,
  - the round's RoPE cos/sin for [visible context; block] (the policy's contiguous positions),
  - the valid-slot target logits in bf16 (= what the per-round path computes, head(h*) in bf16, before .float()).
Rounds are padded to a common length and batched; padded keys are masked out (bool mask, gqa_attention), padded rows
of the new context get zero attention weight and hence zero gradient. The summed loss equals the per-round losses
summed, so the gradient equals the per-round accumulated gradient (up to float summation order).
"""
from types import SimpleNamespace

import torch
from dflash.model import rotate_half

import contextlib

import dflash.model as dm

from .window import deferred_key_rope


def gqa_attention(module, q, k, v, attention_mask, scaling=None, **kw):
    """Masked attention without expanding K/V to the query heads (memory: the batched step's main cost was the
    repeat_kv copies + math-SDPA fp32 copies of every pooled round). Query head h uses KV head h // groups, as
    repeat_kv. Scores and softmax in fp32; same maths as SDPA up to float rounding. Returns [B, Q, H, d]."""
    B, H, Q, d = q.shape
    Hkv, K = k.shape[1], k.shape[2]
    g = H // Hkv
    s = (q.float().reshape(B, Hkv, g * Q, d) @ k.float().transpose(-1, -2)).view(B, Hkv, g, Q, K) * scaling
    if attention_mask is not None:  # bool [B, 1, Q, K], True = attend
        s = s.masked_fill(~attention_mask[:, :, None], float("-inf"))
    p = torch.softmax(s, dim=-1).to(v.dtype).view(B, Hkv, g * Q, K)
    out = (p @ v).view(B, H, Q, d)
    return out.transpose(1, 2).contiguous(), None


@contextlib.contextmanager
def _gqa():
    orig = dm.ALL_ATTENTION_FUNCTIONS
    dm.ALL_ATTENTION_FUNCTIONS = {"sdpa": gqa_attention}
    try:
        yield
    finally:
        dm.ALL_ATTENTION_FUNCTIONS = orig


def capture(cache, th_rows, noise, pos, target_logits_bf16, m):
    """Record one round's loss inputs (call after drafting, before the frontier moves past the next round)."""
    c = cache
    if c.topk is not None or getattr(c, "bias", None) is not None:
        raise ValueError("batched update supports plain window / far-chunk contexts only (no top-k, no bias)")
    n = c.n_new
    vis = c.visible_positions()
    tail = torch.arange(c.c0, c.start, device=c.device)
    if vis.numel() < n or not torch.equal(vis[vis.numel() - n:], tail):
        raise ValueError("the round's new context rows must be the tail of the visible set")
    q = noise.shape[1]
    return SimpleNamespace(cache=c, th=th_rows.detach(), noise=noise.detach(), qpos=pos[0, -q:].clone(),
                           old=vis[:vis.numel() - n].clone(), cos=c.cos, sin=c.sin, n=n, q=q,
                           tl=target_logits_bf16.detach(), m=m)


class _BatchView:
    """Cache stand-in for the batched forward: per batch row b, keys = [stale visible rows (buffers, no grad);
    the round's new context rows (graph); block (graph)], rotated with that round's cos/sin, right-padded."""

    def __init__(self, recs, N, K):
        self.recs, self.N, self.K = recs, N, K

    def update(self, k, v, layer_idx, cache_kwargs=None):
        N = self.N
        ks, vs = [], []
        for b, r in enumerate(self.recs):
            ok = r.cache.k[layer_idx].index_select(2, r.old)
            ov = r.cache.v[layer_idx].index_select(2, r.old)
            kb = torch.cat([ok, k[b:b + 1, :, :r.n], k[b:b + 1, :, N:]], dim=2)
            vb = torch.cat([ov, v[b:b + 1, :, :r.n], v[b:b + 1, :, N:]], dim=2)
            kb = (kb * r.cos) + (rotate_half(kb) * r.sin)
            pad = self.K - kb.shape[2]
            ks.append(torch.nn.functional.pad(kb, (0, 0, 0, pad)))
            vs.append(torch.nn.functional.pad(vb, (0, 0, 0, pad)))
        return torch.cat(ks, 0), torch.cat(vs, 0)


def batched_loss(drafter, head, recs, weights):
    """Summed loss of `recs` (all with the same block size) from one batched forward. Returns the loss tensor."""
    q = recs[0].q
    assert all(r.q == q for r in recs)
    dev = recs[0].noise.device
    N = max(r.n for r in recs)
    lens = [r.old.numel() + r.n + q for r in recs]
    K = max(lens)
    F = recs[0].th.shape[-1]
    th = torch.zeros((len(recs), N, F), dtype=recs[0].th.dtype, device=dev)
    for b, r in enumerate(recs):
        th[b, :r.n] = r.th[0]
    noise = torch.cat([r.noise for r in recs], 0)
    # only the block's ids matter (queries are rotated with the last q positions; keys are rotated in _BatchView)
    pos = torch.cat([torch.zeros((len(recs), N), dtype=torch.long, device=dev),
                     torch.stack([r.qpos for r in recs])], 1)
    mask = torch.zeros((len(recs), 1, q, K), dtype=torch.bool, device=dev)
    for b, L in enumerate(lens):
        mask[b, :, :, :L] = True
    with deferred_key_rope(), _gqa():
        out = drafter(target_hidden=th, noise_embedding=noise, position_ids=pos, attention_mask=mask,
                      past_key_values=_BatchView(recs, N, K), use_cache=True)[:, 1 - q:, :]
    z = torch.cat([out[b, :r.m] for b, r in enumerate(recs)], 0)
    w = torch.cat([weights[:r.m] for r in recs])
    q_logp = torch.log_softmax(drafter.compute_logits(z[None], head)[0].float(), dim=-1)
    p_logp = torch.log_softmax(torch.cat([r.tl for r in recs], 0).float(), dim=-1)
    kl = (p_logp.exp() * (p_logp - q_logp)).sum(-1)
    return (w * kl).sum()


def state_bytes(recs):
    return sum(r.th.numel() * r.th.element_size() + r.noise.numel() * r.noise.element_size()
               + r.tl.numel() * r.tl.element_size() + r.old.numel() * 8 for r in recs)
