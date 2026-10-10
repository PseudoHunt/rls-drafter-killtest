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
                 topk=None, shared=False):
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
        self.rotary = drafter.rotary_emb
        self.dtype, self.device = dtype, device

    def begin(self, c0, start, verify_size):
        """Before a drafter forward whose new context rows are positions [c0, start). Returns position_ids."""
        self.c0, self.start, self.n_new = c0, start, start - c0
        self.k_hi = min(self.keep, start)  # kept: [0, k_hi)
        self.w_lo = self.k_hi if self.window is None else max(self.k_hi, start - self.window)  # window: [w_lo, start)
        self.L = self.k_hi + (start - self.w_lo)  # visible context keys
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

    def _visible(self, buf):
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
