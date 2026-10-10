"""Verifier-Prior Attention (rlsdk/window.py `_vpa_bias`, rlsdk/target_attn.py `key_mass`) on the tiny model (CPU)."""
import pytest
import torch
from dflash.model import _draft_value, _make_cache, _raw_input_embeddings

import rlsdk.replay as rp
from rlsdk.adapters import VPA, FarSel, Window
from rlsdk.replay import simulate, teacher_forced
from rlsdk.target_attn import default_spec, key_mass
from test_replay import NEW, P, _generate, _setup

W0 = 40  # tiny-scale stand-in for 1024 (tiny traces are ~300 tokens)


def _spec(t):
    s = default_spec(t)
    return {**s, "qk": s["mid3"]}


def _tf(t, d, ids):
    return teacher_forced(t, ids, d.target_layer_ids, attn_spec=_spec(t))


def _drafts(d, t, ids, tf, ad):
    orig, seen = d.compute_logits, []
    d.compute_logits = lambda hh, head: seen.append(orig(hh, head)) or seen[-1]
    try:
        rounds = simulate(d, t, ids, P, tf, ad, NEW)
    finally:
        d.compute_logits = orig
    return [(r["t"], r["a"], r["tau"], r["ctx_len"]) for r in rounds], seen


def _same(a, b):
    return a[0] == b[0] and len(a[1]) == len(b[1]) > 10 and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_beta_lam_zero_M16_is_S2_bit_exact(dtype):
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    s2 = _drafts(d, t, ids, tf, FarSel({"W": W0, "keep": "prompt", "n": 2, "score": "mid3"}))
    vpa = _drafts(d, t, ids, tf, VPA({"W0": W0, "M": 2, "beta": 0, "lam": 0}))
    assert _same(s2, vpa)
    assert any(r[3] > P + W0 for r in s2[0])  # far chunks really active


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_beta_lam_zero_M0_is_window_bit_exact(dtype):
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    win = _drafts(d, t, ids, tf, Window({"W": W0, "keep": "prompt"}))
    vpa = _drafts(d, t, ids, tf, VPA({"W0": W0, "M": 0, "beta": 0, "lam": 0}))
    assert _same(win, vpa)


@pytest.mark.parametrize("beta,lam", [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0)])
def test_constant_offset_changes_nothing(beta, lam):
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    a = _drafts(d, t, ids, tf, VPA({"W0": W0, "M": 2, "beta": beta, "lam": lam}))
    b = _drafts(d, t, ids, tf, VPA({"W0": W0, "M": 2, "beta": beta, "lam": lam, "b_offset": 7.5}))
    assert a[0] == b[0]
    assert all(torch.allclose(x, y, atol=1e-4) for x, y in zip(a[1], b[1]))  # fp32 rounding of the centring
    base = _drafts(d, t, ids, tf, VPA({"W0": W0, "M": 2, "beta": 0, "lam": 0}))
    assert not all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a[1], base[1]))  # the bias does act


def test_bias_terms_and_block_keys_never_biased(monkeypatch):
    """Bias vector: centred over kept context keys; distance term only on far-chunk keys; the attention mask has
    exactly 0 on every block key; the biased forward equals a fresh forward with the same additive mask."""
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    beta, lam = 0.7, 0.9
    seen_masks, states = [], []
    import rlsdk.window as w
    sdpa = w._orig_attn["sdpa"]

    def spy_sdpa(module, q, k, v, mask, **kw):
        seen_masks.append((mask, k.shape[2]))
        return sdpa(module, q, k, v, mask, **kw)

    monkeypatch.setattr(w, "_orig_attn", {"sdpa": spy_sdpa})
    orig_begin = rp.WindowCache.begin

    def begin(self, c0, start, vs):
        out = orig_begin(self, c0, start, vs)
        states.append((start, self.visible_positions().clone(), None if self.bias is None else self.bias.clone(),
                       self.k_hi, self.w_lo))
        return out

    monkeypatch.setattr(rp.WindowCache, "begin", begin)
    zs = []

    class Spy(VPA):
        def apply(self, z, slots):
            zs.append(z.clone())
            return z

    simulate(d, t, ids, P, tf, Spy({"W0": W0, "M": 2, "beta": beta, "lam": lam}), NEW)
    n_layers = len(d.layers)
    scale = float(_draft_value(d.config, "input_embedding_scale", 1.0))
    checked = 0
    for r, (start, pos, b, k_hi, w_lo) in enumerate(states):
        mask, K = seen_masks[r * n_layers]
        L = pos.numel()
        assert torch.equal(mask[0, 0, :, L:], torch.zeros_like(mask[0, 0, :, L:]))  # block keys: 0
        assert abs(float(b.mean())) < 1e-6  # centred over context keys
        m = key_mass(tf.qk["q"][start - 1], tf.qk["k"][:start], tf.qk["scaling"])
        far = (pos >= k_hi) & (pos < w_lo)
        dist = (start - pos).float()
        ref = beta * torch.log(m[pos] + 1 / start) - lam * torch.where(far, torch.log1p((dist - W0).clamp_min(0) / W0), 0 * dist)
        assert torch.allclose(b, ref - ref.mean(), atol=1e-5)
        if start + 16 > ids.shape[0] or not far.any():
            continue
        block = torch.full((1, 16), d.mask_token_id, dtype=torch.long)
        block[0, 0] = ids[start]
        m4 = torch.cat([b, torch.zeros(16)]).view(1, 1, 1, -1).expand(1, 1, 16, -1)
        with torch.no_grad():
            ref_z = d(target_hidden=tf.ctx[pos][None], noise_embedding=_raw_input_embeddings(t, block, scale),
                      position_ids=torch.arange(0, L + 16)[None], past_key_values=_make_cache(d.config),
                      use_cache=True, attention_mask=m4)[:, -15:, :][0]
        assert torch.allclose(zs[r], ref_z, atol=1e-5), (start, (zs[r] - ref_z).abs().max())
        checked += 1
    assert checked > 20


def test_key_mass_matches_eager_attention():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    t.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            att = t(ids[None], output_attentions=True).attentions
    finally:
        t.config._attn_implementation = "sdpa"
    layers = tf.qk["layers"]
    for row in (P - 1, 100, ids.shape[0] - 1):
        ref = sum(att[l][0, :, row, :row + 1].sum(0) for l in layers)
        ref = ref / ref.sum()
        got = key_mass(tf.qk["q"][row], tf.qk["k"][:row + 1], tf.qk["scaling"])
        assert torch.allclose(got, ref, atol=1e-5)


def test_vpa_reads_only_up_to_last_committed(monkeypatch):
    from rlsdk.guard import Guarded
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    reads = []

    class Rec(Guarded):
        def __getitem__(self, idx):
            if self.name.startswith("qk"):
                hi = (idx.stop - 1) if isinstance(idx, slice) else idx
                reads.append((self.name, hi, self._f.t))
            return super().__getitem__(idx)

    monkeypatch.setattr(rp, "Guarded", Rec)
    simulate(d, t, ids, P, tf, VPA({"W0": W0, "M": 2, "beta": 1, "lam": 1}), NEW)
    assert reads and all(hi == f - 1 for _, hi, f in reads)
