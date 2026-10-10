"""Context policies B/E/C/C'/D (rlsdk/window.py, NOTES.md §10) on the tiny model (CPU)."""
import pytest
import torch
from dflash.model import _draft_value, _make_cache, _raw_input_embeddings

import dflash.model as dm
from rlsdk.adapters import Baseline, CtxPolicy
from rlsdk.replay import simulate, teacher_forced
from rlsdk.window import topk_mask
from test_replay import NEW, P, _generate, _setup

BIG = 10**6  # >= any context length here


def _drafts(d, t, ids, tf, ad):
    orig, seen = d.compute_logits, []
    d.compute_logits = lambda hh, head: seen.append(orig(hh, head)) or seen[-1]
    try:
        rounds = simulate(d, t, ids, P, tf, ad, NEW)
    finally:
        d.compute_logits = orig
    return [(r["t"], r["a"], r["tau"]) for r in rounds], seen


def _same(a, b):
    (ra, la), (rb, lb) = a, b
    return ra == rb and len(la) == len(lb) and all(torch.equal(x, y) for x, y in zip(la, lb))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("method,cfg", [
    ("winorig", {"W": BIG, "keep": "prompt"}),
    ("capctx", {"cap": BIG}),
    ("topk", {"k": BIG}),
    ("topkcap", {"k": BIG, "cap": BIG}),
    ("topkshared", {"k": BIG, "cap": BIG}),
])
def test_no_op_settings_reproduce_baseline_bit_exact(dtype, method, cfg):
    """cap >= context and k >= context (and W >= context) -> every round's draft logits equal the baseline's."""
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    assert _same(_drafts(d, t, ids, tf, Baseline()), _drafts(d, t, ids, tf, CtxPolicy(method, cfg)))
    assert dm.ALL_ATTENTION_FUNCTIONS is __import__("rlsdk.window", fromlist=["x"])._orig_attn  # patch restored


@pytest.mark.parametrize("method", ["topkcap", "topkshared"])
def test_topk_ge_context_equals_cap_only(method):
    """k >= context under a binding cap: identical to E (capctx) with the same cap."""
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    e = _drafts(d, t, ids, tf, CtxPolicy("capctx", {"cap": 40}))
    assert _same(e, _drafts(d, t, ids, tf, CtxPolicy(method, {"k": BIG, "cap": 40})))
    assert not _same(e, _drafts(d, t, ids, tf, Baseline()))


def _fresh(d, t, ids, tf, rows, kpos, s):
    """Plain drafter forward (stock cache) over context rows `rows` rotated at `kpos`, block at s .. s+15."""
    scale = float(_draft_value(d.config, "input_embedding_scale", 1.0))
    block = torch.full((1, 16), d.mask_token_id, dtype=torch.long)
    block[0, 0] = ids[s]
    pos = torch.cat([kpos, torch.arange(s, s + 16)])[None]
    with torch.no_grad():
        return d(target_hidden=tf.ctx[rows][None], noise_embedding=_raw_input_embeddings(t, block, scale),
                 position_ids=pos, past_key_values=_make_cache(d.config), use_cache=True)[:, -15:, :][0]


@pytest.mark.parametrize("method,cfg", [("winorig", {"W": 40, "keep": "prompt"}), ("capctx", {"cap": 40})])
def test_active_policy_equals_fresh_forward(method, cfg):
    """B: visible rows = prompt + last W, original positions. E: all rows, positions max(p, s - cap)."""
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    zs = []

    class Spy(CtxPolicy):
        def apply(self, z, slots):
            zs.append(z.clone())
            return z

    rounds = simulate(d, t, ids, P, tf, Spy(method, cfg), NEW)
    checked = 0
    for r, z in zip(rounds, zs):
        s = r["t"]
        if s <= P + 40 or s + 16 > ids.shape[0]:
            continue
        if method == "winorig":
            rows = torch.cat([torch.arange(0, P), torch.arange(s - 40, s)])
            kpos = rows
        else:
            rows = torch.arange(0, s)
            kpos = rows.clamp_min(s - 40)
        ref = _fresh(d, t, ids, tf, rows, kpos, s)
        assert torch.allclose(z, ref, atol=1e-5, rtol=1e-5), (s, (z - ref).abs().max())
        checked += 1
    assert checked > 20


@pytest.mark.parametrize("shared", [False, True])
def test_topk_mask_selects_top_scores(shared):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 8, 5, 16, generator=g)  # 8 query heads, 2 kv heads (GQA groups of 4)
    kc = torch.randn(1, 2, 30, 16, generator=g)
    m = topk_mask(q, kc, 0.25, 7, shared)
    kr = kc.repeat_interleave(4, dim=1)  # transformers repeat_kv layout
    s = (q @ kr.transpose(-1, -2)) * 0.25
    if shared:
        s = s.sum(2, keepdim=True).expand_as(s)
    ref = torch.zeros_like(m).scatter_(-1, s.topk(7, dim=-1).indices, True)
    assert torch.equal(m, ref) and int(m.sum(-1).unique()) == 7
    if shared:
        assert all(torch.equal(m[:, :, 0], m[:, :, i]) for i in range(5))


def test_topk_attention_is_renormalised_masked_softmax():
    """The patched attention = softmax over (top-k context keys + all block keys), renormalised, times V."""
    import rlsdk.window as w
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 8, 4, 16, generator=g)
    k = torch.randn(1, 2, 30 + 4, 16, generator=g)
    v = torch.randn(1, 2, 30 + 4, 16, generator=g)

    class Mod(torch.nn.Module):
        num_key_value_groups, is_causal, training = 4, False, False

    cache = type("C", (), {"L": 30, "topk": 6, "shared": False})()
    with w.topk_attention(cache):
        out, _ = dm.ALL_ATTENTION_FUNCTIONS["sdpa"](Mod(), q, k, v, None, scaling=0.25, dropout=0.0)
    kr, vr = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    s = (q @ kr.transpose(-1, -2)) * 0.25
    keep = torch.ones_like(s, dtype=torch.bool)
    keep[..., :30] = torch.zeros_like(s[..., :30], dtype=torch.bool).scatter_(-1, s[..., :30].topk(6, -1).indices, True)
    ref = (torch.softmax(s.masked_fill(~keep, float("-inf")), -1) @ vr).transpose(1, 2)
    assert torch.allclose(out, ref, atol=1e-5)


def test_selection_changes_drafts_when_active():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    base = _drafts(d, t, ids, tf, Baseline())
    for method, cfg in (("topk", {"k": 16}), ("topkcap", {"k": 16, "cap": 40}), ("topkshared", {"k": 16, "cap": 40})):
        assert not _same(base, _drafts(d, t, ids, tf, CtxPolicy(method, cfg)))
