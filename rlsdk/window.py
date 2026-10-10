"""Windowed drafter context, StreamingLLM-style (training-free). See NOTES.md §9.

The drafter attends to a kept set (context positions 0 .. keep-1) plus the last W committed context positions, with
contiguous position ids inside that visible set: kept -> 0 .. k-1, window -> k .. L-1, block -> L .. L+15. The
largest query-key distance is L + 15 <= keep + W + 15. Target pass and verification are untouched.

DFlash rotates context keys before `past_key_values.update`, so the stock cache holds keys rotated at absolute
positions. Here, inside `deferred_key_rope()`, the attention's RoPE rotates queries only and `WindowCache.update`
receives pre-RoPE keys: it stores the new context rows, then rotates [visible context; block] at the contiguous
positions with the same expression as dflash. When the whole context is visible the positions are the absolute ones
and every tensor equals the baseline's element for element.
"""
import contextlib

import torch
import dflash.model as dm
from dflash.model import rotate_half

_orig_apply = dm.apply_rotary_pos_emb


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


class WindowCache:
    """Drop-in for the drafter cache: pre-RoPE context K/V for all positions, visible set re-rotated each round."""

    def __init__(self, drafter, keep, window, capacity, device, dtype):
        cfg = drafter.config
        kvh, hd = cfg.num_key_value_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        shape = (1, kvh, capacity, hd)
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in drafter.layers]
        self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in drafter.layers]
        self.keep, self.window = int(keep), int(window)
        self.rotary = drafter.rotary_emb
        self.dtype, self.device = dtype, device

    def begin(self, c0, start, verify_size):
        """Before a drafter forward whose new context rows are positions [c0, start). Returns position_ids."""
        self.c0, self.start, self.n_new = c0, start, start - c0
        self.k_hi = min(self.keep, start)  # kept: [0, k_hi)
        self.w_lo = max(self.k_hi, start - self.window)  # window: [w_lo, start)
        self.L = self.k_hi + (start - self.w_lo)
        pos = torch.arange(0, self.L + verify_size, device=self.device)[None]
        cos, sin = self.rotary(torch.empty(0, dtype=self.dtype, device=self.device), pos)
        self.cos, self.sin = cos.unsqueeze(1), sin.unsqueeze(1)
        # only the block's ids matter (queries); ids of new context rows are unused because keys are rotated here
        return pos[:, self.L - self.n_new:] if self.n_new <= self.L else \
            torch.cat([pos.new_zeros(1, self.n_new - self.L), pos], dim=1)

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
        kk = (kk * self.cos) + (rotate_half(kk) * self.sin)  # dflash's key rotation, contiguous positions
        return kk, vv
