"""Offline replay of DFlash greedy speculative decoding over a fixed greedy trace (brief §4).

Mirrors dflash.model.dflash_generate (dflash==0.1.0) round by round, reusing its drafter forward,
cache helpers and logits path.  Target-side quantities come from one teacher-forced pass and are
only readable below the current frontier (Guarded).
"""
import time
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from dflash.model import (
    _crop_to,
    _make_cache,
    _output_head,
    _raw_input_embeddings,
    _draft_value,
    rotate_half,
)

from .guard import Frontier, Guarded
from .window import WindowCache, deferred_key_rope


# ---------------------------------------------------------------------------
# Teacher-forced target pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_forced(target, ids, layer_ids, chunk=2048):
    """ids: 1-D LongTensor (prompt + continuation).

    Returns ctx [T, n*H] (exactly extract_context_feature's layout: outputs of decoder layers
    `layer_ids`, concatenated), hstar [T, H] (final-norm output = lm_head input), argmax [T].
    """
    captured = {}
    hooks = []
    for lid in layer_ids:
        layer = target.model.layers[lid]
        hooks.append(layer.register_forward_hook(
            lambda mod, inp, out, lid=lid: captured.__setitem__(lid, out[0] if isinstance(out, tuple) else out)))
    try:
        out = target.model(input_ids=ids[None], use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    ctx = torch.cat([captured[lid][0] for lid in layer_ids], dim=-1)
    hstar = out.last_hidden_state[0]
    head = _output_head(target)
    argmax = torch.cat([head(hstar[i:i + chunk]).argmax(-1) for i in range(0, hstar.shape[0], chunk)])
    return SimpleNamespace(ctx=ctx, hstar=hstar, argmax=argmax)


def near_tie_rate(argmax, ids, prompt_len):
    """Fraction of continuation-predicting positions j in [P-1, T-2] with argmax_j != s_{j+1}."""
    j = torch.arange(prompt_len - 1, ids.shape[0] - 1, device=ids.device)
    if j.numel() == 0:
        return 0.0
    return float((argmax[j] != ids[j + 1]).float().mean())


# ---------------------------------------------------------------------------
# Drafter helpers
# ---------------------------------------------------------------------------

class PrefixCache:
    """Read-only view of a DynamicCache prefix; update() returns cat(prefix, new) without storing.
    Used for the TTS gradient forward so the shared drafter cache is never touched."""

    def __init__(self, cache, length):
        self.kv = [(layer.keys[..., :length, :], layer.values[..., :length, :]) for layer in cache.layers]

    def update(self, k, v, layer_idx, cache_kwargs=None):
        pk, pv = self.kv[layer_idx]
        return torch.cat([pk, k], dim=-2), torch.cat([pv, v], dim=-2)


@torch.no_grad()
def rebuild_context_cache(drafter, cache, ctx_rows, chunk=4096):
    """Recompute the context K/V for positions [0, len(ctx_rows)) with the drafter's current params,
    replacing the cache contents (same math as Qwen3DFlashAttention.forward for the ctx part)."""
    T = ctx_rows.shape[0]
    ks = [[] for _ in drafter.layers]
    vs = [[] for _ in drafter.layers]
    for s in range(0, T, chunk):
        th = drafter.hidden_norm(drafter.fc(ctx_rows[s:s + chunk][None]))
        pos = torch.arange(s, s + th.shape[1], device=th.device)[None]
        cos, sin = drafter.rotary_emb(th, pos)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        for li, layer in enumerate(drafter.layers):
            attn = layer.self_attn
            k = attn.k_proj(th).view(1, th.shape[1], -1, attn.head_dim)
            k = attn.k_norm(k).transpose(1, 2)
            v = attn.v_proj(th).view(1, th.shape[1], -1, attn.head_dim).transpose(1, 2)
            ks[li].append(k * cos + rotate_half(k) * sin)
            vs[li].append(v)
    for li, layer in enumerate(cache.layers):
        layer.keys = torch.cat(ks[li], dim=-2)
        layer.values = torch.cat(vs[li], dim=-2)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Round simulation
# ---------------------------------------------------------------------------

def simulate(drafter, target, ids, prompt_len, tf, adapter, max_new_tokens, block_size=None,
             on_round=None, round_hook=None, max_rounds=None):
    """Replay DFlash greedy rounds over the committed trace `ids` (1-D, prompt + continuation).

    Returns list of per-round dicts.  `round_hook(rnd)` (tests) is called after verification with the
    guarded arrays, before the adapter update.
    """
    device = ids.device
    block_size = drafter.block_size if block_size is None else block_size
    P, T = prompt_len, ids.shape[0]
    end = T - 1  # last committed index of the trace
    max_length = P + max_new_tokens
    head = _output_head(target)
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    mask_id = drafter.mask_token_id
    H = drafter.config.hidden_size

    frontier = Frontier(P)  # prefill revealed target states for positions < P
    ctx = Guarded(tf.ctx, frontier, "ctx")
    hstar = Guarded(tf.hstar, frontier, "hstar")
    win = getattr(adapter, "ctx_window", None)  # windowed drafter context (rlsdk/window.py)
    if win is None:
        cache = _make_cache(drafter.config)
    else:
        keep = P if win["keep"] == "prompt" else int(win["keep"])
        cache = WindowCache(drafter, keep, win["W"], T, device, next(drafter.parameters()).dtype)
    if hasattr(adapter, "bind_head"):  # KL arms: W = lm_head weight
        adapter.bind_head(head)
    adapter.reset(H, device)

    rounds = []
    start, c0 = P, 0  # c0: first context position not yet fed to the drafter
    r = 0
    while start + 1 < max_length and start < end and (max_rounds is None or r < max_rounds):
        verify_size = min(block_size, max_length - start)
        n_slots = verify_size - 1
        t0 = _sync(device)
        if n_slots > 0:
            with torch.no_grad():
                target_hidden = ctx[c0:start][None]
                block = torch.full((1, verify_size), mask_id, dtype=torch.long, device=device)
                block[0, 0] = ids[start]
                noise = _raw_input_embeddings(target, block, scale)
                if win is None:
                    pos = torch.arange(c0, start + verify_size, device=device)[None]
                    z = drafter(target_hidden=target_hidden, noise_embedding=noise, position_ids=pos,
                                past_key_values=cache, use_cache=True)[:, 1 - verify_size:, :][0]
                    _crop_to(cache, start)
                else:
                    pos = cache.begin(c0, start, verify_size)
                    with deferred_key_rope():
                        z = drafter(target_hidden=target_hidden, noise_embedding=noise, position_ids=pos,
                                    past_key_values=cache, use_cache=True)[:, 1 - verify_size:, :][0]
                slots = torch.arange(1, verify_size, device=device)
                zp = adapter.apply(z, slots)
                draft = drafter.compute_logits(zp[None], head)[0].argmax(-1)
        draft_ms = (_sync(device) - t0) * 1e3

        # --- verification (greedy): accept the longest prefix matching the committed trace
        if n_slots > 0:
            truth = ids[start + 1:min(start + verify_size, T)]
            match = (draft[:truth.shape[0]] == truth).long()
            a = int(match.cumprod(0).sum())
        else:
            a = 0
        produced = min(a + 1, max_length - start - 1, end - start)
        t, c0 = start, start
        start += produced
        frontier.advance(start)  # target states revealed on the true prefix up to t + a

        # --- adaptation on valid pairs (z_i, h*_{t+i-1}), i = 1..m
        update_ms = 0.0
        m = min(produced, n_slots)
        rnd = None
        if m > 0:
            rnd = SimpleNamespace(ctx=ctx, hstar=hstar, frontier=frontier, t=t, m=m)
            if round_hook is not None:
                round_hook(rnd)
            t1 = _sync(device)
            h = hstar[t:t + m]
            if adapter.method == "tts":
                th_rows = target_hidden
                prefix_len = t - th_rows.shape[1]

                def grad_forward():
                    # math SDPA: deterministic backward (fused SDPA kernels accumulate dQ/dK with atomics)
                    with sdpa_kernel(SDPBackend.MATH):
                        return drafter(target_hidden=th_rows, noise_embedding=noise, position_ids=pos,
                                       past_key_values=PrefixCache(cache, prefix_len), use_cache=True,
                                       )[:, 1 - verify_size:, :][0]

                rnd.grad_forward = grad_forward
                rnd.draft_logits = lambda zz: drafter.compute_logits(zz[None], head)[0]
                with torch.no_grad():
                    rnd.target_logits = head(h).float()
                rnd.refresh_cache = lambda: rebuild_context_cache(drafter, cache, ctx[0:t])
            adapter.update(z[:m], h, slots[:m], rnd)
            update_ms = (_sync(device) - t1) * 1e3

        # solve_ms: RLS linear-algebra time this round (Cholesky / solves); contained in draft_ms + update_ms
        solve_ms = adapter.pop_solve_ms() if hasattr(adapter, "pop_solve_ms") else 0.0
        row = {"round": r, "t": t, "pos": t - P, "a": a, "tau": produced, "m": m,
               "draft_ms": round(draft_ms, 4), "update_ms": round(update_ms, 4), "solve_ms": round(solve_ms, 4)}
        if win is not None:
            row["ctx_len"] = cache.L  # visible context rows; max query-key distance = ctx_len + n_slots
        if hasattr(adapter, "pop_round_log"):
            row.update(adapter.pop_round_log())
        rounds.append(row)
        if on_round is not None:
            on_round(row)
        r += 1
    return rounds
