"""Windowed drafter context (rlsdk/window.py) on the tiny model (CPU)."""
import pytest
import torch
from dflash.model import _make_cache, _raw_input_embeddings, _draft_value

import dflash.model as dm
from rlsdk.adapters import Baseline, Window
from rlsdk.replay import simulate, teacher_forced
from test_replay import NEW, P, _generate, _setup


def _drafts(d, t, ids, tf, ad):
    orig, seen = d.compute_logits, []
    d.compute_logits = lambda hh, head: seen.append(orig(hh, head)) or seen[-1]
    try:
        rounds = simulate(d, t, ids, P, tf, ad, NEW)
    finally:
        d.compute_logits = orig
    return rounds, seen


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keep,W", [(64, P + NEW), ("prompt", NEW), (8, 10**6)])
def test_context_within_window_is_bit_exact(dtype, keep, W):
    """Context never longer than kept set + window -> every round's draft logits equal the baseline's bit for bit."""
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    rb, lb = _drafts(d, t, ids, tf, Baseline())
    rw, lw = _drafts(d, t, ids, tf, Window({"W": W, "keep": keep}))
    assert len(lb) == len(lw) > 10
    assert all(torch.equal(a, b) for a, b in zip(lb, lw))
    assert [(r["t"], r["a"], r["tau"]) for r in rb] == [(r["t"], r["a"], r["tau"]) for r in rw]
    assert all(r["ctx_len"] == r["t"] for r in rw)  # whole context visible at absolute positions
    assert dm.apply_rotary_pos_emb is dm.__dict__["apply_rotary_pos_emb"] and \
        dm.apply_rotary_pos_emb.__name__ == "apply_rotary_pos_emb"  # patch restored


@pytest.mark.parametrize("keep", [8, "prompt"])
def test_window_equals_fresh_forward_on_visible_rows(keep):
    """With the window active, each round's z equals a plain drafter forward (stock cache) over only the visible
    context rows at contiguous positions 0..L-1 and the block at L..L+15; L <= keep + W."""
    W = 40
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    zs = []

    class Spy(Window):
        def apply(self, z, slots):
            zs.append(z.clone())
            return z

    rounds = simulate(d, t, ids, P, tf, Spy({"W": W, "keep": keep}), NEW)
    k = P if keep == "prompt" else keep
    scale = float(_draft_value(d.config, "input_embedding_scale", 1.0))
    checked = 0
    for r, z in zip(rounds, zs):
        s = r["t"]
        assert r["ctx_len"] == min(s, min(k, s) + W) <= k + W
        if s <= k + W or s + 16 > ids.shape[0]:
            continue
        vis = torch.cat([tf.ctx[:k], tf.ctx[s - W:s]])
        block = torch.full((1, 16), d.mask_token_id, dtype=torch.long)
        block[0, 0] = ids[s]
        with torch.no_grad():
            ref = d(target_hidden=vis[None], noise_embedding=_raw_input_embeddings(t, block, scale),
                    position_ids=torch.arange(0, k + W + 16)[None], past_key_values=_make_cache(d.config),
                    use_cache=True)[:, -15:, :][0]
        assert torch.allclose(z, ref, atol=1e-5, rtol=1e-5), (s, (z - ref).abs().max())
        checked += 1
    assert checked > 20


def test_window_changes_drafts_when_active():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    _, lb = _drafts(d, t, ids, tf, Baseline())
    _, lw = _drafts(d, t, ids, tf, Window({"W": 16, "keep": 4}))
    assert any(not torch.equal(a, b) for a, b in zip(lb, lw))
