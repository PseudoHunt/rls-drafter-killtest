"""Drafter context policies (training-free): windowing, position remapping, top-k key selection. NOTES.md §9–§10.

Target pass and verification are untouched; only the drafter's attention over the committed context changes.

DFlash rotates context keys before `past_key_values.update`, so the stock cache holds keys rotated at absolute
positions. Here, inside `deferred_key_rope()`, the attention's RoPE rotates queries only and `WindowCache.update`
receives pre-RoPE keys: it stores the new context rows, then rotates [visible context; block] at the policy's key
positions with the same expression as dflash.

Visible set: kept set (positions 0 .. keep-1) + last W context positions (W = None: every position).
Key positions (`positions`):
  contiguous  kept -> 0 .. k-1, window -> k .. L-1, block (queries and block keys) -> L .. L+15  (StreamingLLM-style)
  original    every key at its absolute position, block at start .. start+15
  capped      as original, but keys older than start - cap all sit at start - cap (ReRoPE-style clipping)
Far chunks (`far`, T0b, NOTES.md §11): besides kept + window, keep n aligned 32-token chunks lying wholly in the
far region [k_hi, w_lo), chosen each round by a score row (target-guided: the target's head-summed attention row at
the last committed position start-1, read through the leakage guard; or random with a fixed seed). Visible keys are
kept in original order and renumbered contiguously (positions must be "contiguous").
Selection (`topk`, inside `topk_attention()`): per layer, per query head and per query, keep the top-k context keys
by attention logit under the policy's rotation (block keys always kept) and renormalise (masked softmax);
`shared=True` sums the logits over the block's queries per head and uses one set for the whole block.

When nothing is dropped, moved or masked (whole context visible, cap >= context, k >= context) every tensor equals
the baseline's element for element.
"""
import contextlib

import torch
import dflash.model as dm
from dflash.model import rotate_half

_orig_apply = dm.apply_rotary_pos_emb
_orig_attn = dm.ALL_ATTENTION_FUNCTIONS


def _rope_q_only(q, k, cos, sin, unsqueeze_dim=1):
    """dflash.model.apply_rotary_pos_emb with the key rotation left to WindowCache.update."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    return q_embed, k


@contextlib.contextmanager
def deferred_key_rope():
    dm.apply_rotary_pos_emb = _rope_q_only
    try:
        yield
    finally:
        dm.apply_rotary_pos_emb = _orig_apply


def topk_mask(q, kc, scaling, k, shared=False):
    """Bool [b, Hq, Q, n]: True for the top-k context keys per (query head, query) by logit q.k*scaling (fp32);
    shared: per head, top-k of the logits summed over the Q queries, one set for every query. GQA: query head h
    uses kv head h // (Hq/Hkv), as transformers' repeat_kv."""
    b, Hq, Q, d = q.shape
    Hkv, n = kc.shape[1], kc.shape[2]
    s = (q.float().reshape(b, Hkv, (Hq // Hkv) * Q, d) @ kc.float().transpose(-1, -2)).reshape(b, Hq, Q, n) * scaling
    if shared:
        s = s.sum(2, keepdim=True)
    idx = s.topk(k, dim=-1).indices
    mask = torch.zeros(s.shape, dtype=torch.bool, device=q.device).scatter_(-1, idx, True)
    return mask.expand(b, Hq, Q, n)


@contextlib.contextmanager
def topk_attention(cache):
    sdpa = _orig_attn["sdpa"]

    def attn(module, q, k, v, attention_mask, **kw):
        n = cache.L
        if getattr(cache, "bias", None) is not None:  # VPA: additive bias on context keys only, 0 on block keys
            b = torch.cat([cache.bias, cache.bias.new_zeros(k.shape[2] - n)]).to(q.dtype)
            return sdpa(module, q, k, v, b.view(1, 1, 1, -1).expand(1, 1, q.shape[2], -1), **kw)
        if cache.topk is None or cache.topk >= n:
            return sdpa(module, q, k, v, attention_mask, **kw)
        m = topk_mask(q, k[:, :, :n], kw["scaling"], cache.topk, cache.shared)
        m = torch.cat([m, m.new_ones(m.shape[:-1] + (k.shape[2] - n,))], dim=-1)  # block keys always visible
        return sdpa(module, q, k, v, m, **kw)

    dm.ALL_ATTENTION_FUNCTIONS = {"sdpa": attn}
    try:
        yield
    finally:
        dm.ALL_ATTENTION_FUNCTIONS = _orig_attn


class WindowCache:
    """Drop-in for the drafter cache: pre-RoPE context K/V for all positions, visible set rotated each round."""

    def __init__(self, drafter, keep, window, capacity, device, dtype, positions="contiguous", cap=None,
                 topk=None, shared=False, far=None):
        cfg = drafter.config
        kvh, hd = cfg.num_key_value_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        shape = (1, kvh, capacity, hd)
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in drafter.layers]
        self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in drafter.layers]
        self.keep = int(keep or 0)
        self.window = None if window is None else int(window)
        assert positions in ("contiguous", "original", "capped") and (positions != "capped" or cap is not None)
        self.positions, self.cap = positions, None if cap is None else int(cap)
        self.topk = None if topk is None else int(topk)
        self.shared = bool(shared)
        # far = {"n": chunks, "chunk": 32, "score": Guarded [T, nC] row source or None (random), "seed": int}
        self.far = far
        assert far is None or positions == "contiguous"
        if far is not None and far.get("score") is None:
            self.rng = torch.Generator(device="cpu").manual_seed(int(far.get("seed", 0)))
        self.idx = None
        self.far_chunks = []  # chunk ids kept this round (logged)
        self.rotary = drafter.rotary_emb
        self.dtype, self.device = dtype, device

    def begin(self, c0, start, verify_size):
        """Before a drafter forward whose new context rows are positions [c0, start). Returns position_ids."""
        self.c0, self.start, self.n_new = c0, start, start - c0
        self.k_hi = min(self.keep, start)  # kept: [0, k_hi)
        self.w_lo = self.k_hi if self.window is None else max(self.k_hi, start - self.window)  # window: [w_lo, start)
        self.L = self.k_hi + (start - self.w_lo)  # visible context keys
        self.idx, self.far_chunks = None, []
        self.bias = None
        if self.far is not None:
            self._select_far(start)
            if self.far.get("vpa"):
                self._vpa_bias(start)
        ar = lambda a, b: torch.arange(a, b, device=self.device)  # noqa: E731
        if self.positions == "contiguous":
            kpos, qpos = ar(0, self.L), ar(self.L, self.L + verify_size)
        else:
            kpos = ar(0, start) if self.w_lo == self.k_hi else torch.cat([ar(0, self.k_hi), ar(self.w_lo, start)])
            if self.positions == "capped":
                kpos = kpos.clamp_min(start - self.cap)
            qpos = ar(start, start + verify_size)
        self.kpos = kpos
        cos, sin = self.rotary(torch.empty(0, dtype=self.dtype, device=self.device), torch.cat([kpos, qpos])[None])
        self.cos, self.sin = cos.unsqueeze(1), sin.unsqueeze(1)
        # only the block's ids matter (queries); ids of the new context rows are unused (keys are rotated here)
        return torch.cat([qpos.new_zeros(self.n_new), qpos])[None]

    def _select_far(self, start):
        C, n = int(self.far.get("chunk", 32)), int(self.far["n"])
        lo, hi = -(-self.k_hi // C), self.w_lo // C  # chunk ids wholly inside [k_hi, w_lo)
        if hi <= lo:
            return  # no far context: identical to the plain window
        cand = torch.arange(lo, hi)
        if self.far.get("score") is None:
            pick = cand[torch.randperm(cand.numel(), generator=self.rng)[:n]]
        else:
            row = self.far["score"][start - 1]  # guarded: only rows <= last committed position
            sc = row[lo:hi].float().cpu()
            pick = cand[sc.topk(min(n, cand.numel())).indices]
        pick = pick.sort().values
        self.far_chunks = pick.tolist()
        far_pos = (pick[:, None] * C + torch.arange(C)[None]).reshape(-1).to(self.device)
        ar = lambda a, b: torch.arange(a, b, device=self.device)  # noqa: E731
        self.idx = torch.cat([ar(0, self.k_hi), far_pos, ar(self.w_lo, start)])
        self.L = int(self.idx.numel())

    def visible_positions(self):
        if self.idx is not None:
            return self.idx
        return torch.cat([torch.arange(0, self.k_hi, device=self.device),
                          torch.arange(self.w_lo, self.start, device=self.device)])

    def _vpa_bias(self, start):
        """b_j = beta*log(m_j + eps) - lam*log(1 + max(0, d_j - W0)/W0) on kept context keys (distance term on far
        chunk keys only), centred over the kept context keys. m = target attention mass at t'-1 (guarded reads of
        the q row start-1 and K rows < start), eps = 1/N_ctx, d_j = start - p_j. beta = lam = 0 -> no bias at all."""
        v = self.far["vpa"]
        beta, lam, W0 = float(v["beta"]), float(v["lam"]), float(v["W0"])
        if beta == 0.0 and lam == 0.0:
            return
        pos = self.visible_positions()
        b = torch.zeros(pos.numel(), dtype=torch.float32, device=self.device)
        if beta:
            from .target_attn import key_mass
            m = key_mass(v["q"][start - 1], v["k"][0:start], v["scaling"]).to(self.device)
            b += beta * torch.log(m[pos] + 1.0 / start)
        if lam:
            far = (pos >= self.k_hi) & (pos < self.w_lo)  # far-chunk keys (not prompt, not recent window)
            d = (start - pos).float()
            b -= lam * torch.where(far, torch.log1p((d - W0).clamp_min(0) / W0), torch.zeros_like(d))
        b += float(v.get("b_offset", 0.0))  # tests only: a constant must not change anything
        self.bias = b - b.mean()

    def _visible(self, buf):
        if self.idx is not None:
            return buf[:, :, :self.start].index_select(2, self.idx)
        if self.w_lo == self.k_hi:
            return buf[:, :, :self.start]
        return torch.cat([buf[:, :, :self.k_hi], buf[:, :, self.w_lo:self.start]], dim=2)

    def update(self, k, v, layer_idx, cache_kwargs=None):
        n = self.n_new
        self.k[layer_idx][:, :, self.c0:self.start] = k[:, :, :n]
        self.v[layer_idx][:, :, self.c0:self.start] = v[:, :, :n]
        kk = torch.cat([self._visible(self.k[layer_idx]), k[:, :, n:]], dim=2)
        vv = torch.cat([self._visible(self.v[layer_idx]), v[:, :, n:]], dim=2)
        kk = (kk * self.cos) + (rotate_half(kk) * self.sin)  # dflash's key rotation, at the policy's positions
        return kk, vv


class _GradView:
    """Read-only view of a WindowCache for the TTS gradient forward of the current round: the visible keys are the
    ones the round was drafted with (same kept set / far chunks / window, same positions); the round's new context
    rows and the block come from the graph (with grad), older visible rows from the buffers (stale, no grad), as
    `replay.PrefixCache` does for the stock cache. Nothing is written."""

    def __init__(self, cache):
        c = cache
        n = c.n_new
        vis = c.idx if c.idx is not None else torch.cat([torch.arange(0, c.k_hi, device=c.device),
                                                         torch.arange(c.w_lo, c.start, device=c.device)])
        tail = torch.arange(c.c0, c.start, device=c.device)
        if vis.numel() < n or not torch.equal(vis[vis.numel() - n:], tail):
            raise ValueError("the round's new context rows must be the tail of the visible set")
        self.c, self.old = c, vis[:vis.numel() - n]

    def update(self, k, v, layer_idx, cache_kwargs=None):
        c = self.c
        kk = torch.cat([c.k[layer_idx][:, :, :c.start].index_select(2, self.old), k], dim=2)
        vv = torch.cat([c.v[layer_idx][:, :, :c.start].index_select(2, self.old), v], dim=2)
        return (kk * c.cos) + (rotate_half(kk) * c.sin), vv


def grad_view(cache):
    return _GradView(cache)
