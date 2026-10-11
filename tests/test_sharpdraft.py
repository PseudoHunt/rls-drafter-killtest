"""SharpDraft query scaling (arXiv 2610.05106 Eq. 5 / Eq. 18; rlsdk/adapters.py::SharpDraft,
rlsdk/replay.py::scaled_queries) on the tiny model (CPU), plus the S2-layout AIME no-op check on the real traces."""
import glob
import json
import math
import os

import pytest
import torch
from dflash.model import _make_cache, _raw_input_embeddings, _draft_value

import rlsdk.replay as rp
from rlsdk.adapters import Baseline, FarSel, SharpDraft, sharp_scale
from rlsdk.replay import scaled_queries, simulate, teacher_forced
from rlsdk.target_attn import default_spec
from test_replay import NEW, P, _generate, _setup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2_CFG = {"W": 1024, "keep": "prompt", "n": 16, "score": "mid3"}


def _drafts(d, t, ids, tf, ad):
    orig, seen = d.compute_logits, []
    d.compute_logits = lambda hh, head: seen.append(orig(hh, head)) or seen[-1]
    try:
        rounds = simulate(d, t, ids, P, tf, ad, NEW)
    finally:
        d.compute_logits = orig
    return rounds, seen


# --- Eq. 5 ---------------------------------------------------------------------------------------------------------

def test_scale_is_one_at_and_below_reference():
    for N0 in (64, 2048, 4096):
        assert sharp_scale(N0, N0=N0) == 1.0
        for N in (1, 17, N0 // 2, N0 - 1):
            assert sharp_scale(N, N0=N0) == 1.0
        assert sharp_scale(N0 + 1, N0=N0) > 1.0


def test_scale_formula_values():
    # beta = 1 + log((N - k)/(N0 - k))/s
    for N in (2049, 4096, 8192, 16400):
        assert sharp_scale(N) == pytest.approx(1 + math.log((N - 32) / 2016) / 5, rel=1e-12)


def test_cap_reached_and_held():
    cap = math.exp(0.5)
    n_cap = 32 + 2016 * math.exp(5 * (cap - 1))  # smallest N with 1 + log(r)/s >= e^0.5 (~ 51.7K)
    grid = list(range(2000, 200000, 97))
    vals = [sharp_scale(N) for N in grid]
    assert all(b >= a for a, b in zip(vals, vals[1:]))  # non-decreasing
    assert all(v <= cap for v in vals)
    assert sharp_scale(int(n_cap) - 50) < cap
    for N in (int(n_cap) + 1, 10**5, 10**6, 10**9):
        assert sharp_scale(N) == cap  # reached and held
    assert sharp_scale(16400) < 1.45  # our traces (<= 16K output + prompt) never reach the cap


# --- scale forced to 1 = baseline, bit for bit ----------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ctx", ["full", "s2"])
def test_scale_one_is_bit_exact_baseline(dtype, ctx):
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids, attn_spec=default_spec(t))
    ref = Baseline() if ctx == "full" else FarSel(S2_CFG)
    if ctx == "s2":
        ref.ctx_window["W"] = 40  # tiny traces: make the window and far chunks active
        ref.ctx_window["far"]["n"] = 2
    sd = SharpDraft({"N0": 10**9, "ctx": ctx})
    if ctx == "s2":
        sd.ctx_window = json.loads(json.dumps(ref.ctx_window))
    r0, l0 = _drafts(d, t, ids, tf, ref)
    r1, l1 = _drafts(d, t, ids, tf, sd)
    assert [r["tau"] for r in r0] == [r["tau"] for r in r1]
    assert len(l0) == len(l1) and all(torch.equal(a, b) for a, b in zip(l0, l1))
    assert all(r["qscale"] == 1.0 for r in r1)
    if ctx == "s2":
        assert any(r.get("far", 0) > 0 for r in r1)  # far chunks were active


# --- query scaling == logit scaling ---------------------------------------------------------------------------------

def _one_forward(t, d, ids, start):
    """A fresh full-context drafter forward for the block at `start` (fp32 tiny model)."""
    tf = teacher_forced(t, ids, d.target_layer_ids)
    scale = float(_draft_value(d.config, "input_embedding_scale", 1.0))
    block = torch.full((1, 16), d.mask_token_id, dtype=torch.long)
    block[0, 0] = ids[start]
    noise = _raw_input_embeddings(t, block, scale)
    pos = torch.arange(0, start + 16)[None]
    with torch.no_grad():
        return d(target_hidden=tf.ctx[0:start][None], noise_embedding=noise, position_ids=pos,
                 past_key_values=_make_cache(d.config), use_cache=True)[0, -15:]


@pytest.mark.parametrize("beta", [1.07, 1.3, math.exp(0.5)])
def test_query_scaling_equals_logit_scaling(beta):
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    start = P + 100
    base = _one_forward(t, d, ids, start)
    with scaled_queries(d, beta):
        zq = _one_forward(t, d, ids, start)
    # reference: unscaled queries, every attention logit multiplied by beta (softmax scale beta / sqrt(d))
    saved = [layer.self_attn.scaling for layer in d.layers]
    try:
        for layer in d.layers:
            layer.self_attn.scaling = layer.self_attn.scaling * beta
        zl = _one_forward(t, d, ids, start)
    finally:
        for layer, s in zip(d.layers, saved):
            layer.self_attn.scaling = s
    assert torch.allclose(zq, zl, atol=1e-5, rtol=1e-5)
    assert (zq - base).abs().max() > 1e-3  # the scaling is not a no-op
    # hooks are removed on exit
    assert torch.equal(_one_forward(t, d, ids, start), base)


def test_eq18_bf16_rule():
    """Eq. 18: y' = y + cast_bf16(float(y) * (beta - 1)) on the bf16 q_norm output."""
    t, d, _ = _setup(dtype=torch.bfloat16)
    qn = d.layers[0].self_attn.q_norm
    x = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    beta = 1.234
    with scaled_queries(d, beta):
        y2 = qn(x)
    y = qn(x)
    assert torch.equal(y2, y + (y.float() * (beta - 1)).to(torch.bfloat16))


# --- N counting and per-round scale in simulate ---------------------------------------------------------------------

@pytest.mark.parametrize("ctx", ["full", "s2"])
def test_visible_key_count_and_applied_scale(ctx, monkeypatch):
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids, attn_spec=default_spec(t))
    sd = SharpDraft({"N0": 64, "ctx": ctx})  # small reference count: the scale is active on the tiny traces
    if ctx == "s2":
        sd.ctx_window["W"] = 40
        sd.ctx_window["far"]["n"] = 2
    applied = []
    orig = rp.scaled_queries
    monkeypatch.setattr(rp, "scaled_queries", lambda dr, b: applied.append(b) or orig(dr, b))
    rounds = simulate(d, t, ids, P, tf, sd, NEW)
    assert applied == [r["qscale"] for r in rounds]
    for r in rounds:
        vs = min(16, P + NEW - r["t"])
        if ctx == "full":
            assert r["n_vis"] == r["t"] + vs  # all committed positions + block slots
        else:
            assert r["n_vis"] == r["ctx_len"] + vs  # visible context rows + block slots
        assert r["qscale"] == sd.query_scale(r["n_vis"])
    assert any(r["qscale"] > 1.0 for r in rounds)
    if ctx == "s2":
        assert max(r["n_vis"] for r in rounds) <= P + 40 + 2 * 32 + 16


# --- S2 layout on the AIME traces: scale exactly 1 on every round ---------------------------------------------------

def test_s2_layout_aime_scale_is_exactly_one():
    """S2 visible keys <= prompt + 1024 + 16*32 (+ 16 block slots) <= N0 = 2048 for every AIME trace (prompt <= 413),
    so SharpDraft + S2 (+cal) is the S2 (+cal) replay bit for bit on AIME (no hook is registered when beta == 1,
    test_scale_one_is_bit_exact_baseline). Checked structurally on every AIME trace and on the logged visible
    context counts of every AIME round of the saved S2+cal test replays."""
    tp = os.path.join(ROOT, "traces/traces.jsonl")
    if not os.path.exists(tp):
        pytest.skip("traces not present")
    traces = [json.loads(x) for x in open(tp)]
    aime = [tr for tr in traces if tr["dataset"] == "aime"]
    assert len([tr for tr in aime if tr["split"] == "test"]) == 24
    for tr in aime:
        assert sharp_scale(tr["prompt_len"] + 1024 + 16 * 32 + 16) == 1.0, tr["trace_id"]
    files = sorted(glob.glob(os.path.join(ROOT, "results/raw_test_cal/aime*__farsel__W=1024_keep=prompt_n=16_"
                                                "score=mid3.jsonl")))
    if not files:
        pytest.skip("S2+cal replays not present")
    n = 0
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("type") == "round":
                assert sharp_scale(r["ctx_len"] + 16) == 1.0
                n += 1
    assert len(files) >= 23 and n > 10000
