"""Target-guided far-chunk selection (T0b; rlsdk/target_attn.py, rlsdk/window.py far chunks) on the tiny model (CPU)."""
import pytest
import torch
from dflash.model import _draft_value, _make_cache, _raw_input_embeddings

import rlsdk.replay as rp
from rlsdk.adapters import FarSel, Window
from rlsdk.guard import Guarded
from rlsdk.replay import simulate, teacher_forced
from rlsdk.target_attn import default_spec
from test_replay import NEW, P, _generate, _setup


def _tf(t, d, ids):
    return teacher_forced(t, ids, d.target_layer_ids, attn_spec=default_spec(t))


def _drafts(d, t, ids, tf, ad):
    orig, seen = d.compute_logits, []
    d.compute_logits = lambda hh, head: seen.append(orig(hh, head)) or seen[-1]
    try:
        rounds = simulate(d, t, ids, P, tf, ad, NEW)
    finally:
        d.compute_logits = orig
    return rounds, seen


def test_chunk_scores_match_eager_attention():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    t.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            att = t(ids[None], output_attentions=True).attentions
    finally:
        t.config._attn_implementation = "sdpa"
    T = ids.shape[0]
    for name, layers in default_spec(t).items():
        ref = sum(att[l][0].sum(0) for l in layers) / len(layers)  # [T, T], head-summed
        ref = torch.nn.functional.pad(ref, (0, tf.attn[name].shape[1] * 32 - T)).view(T, -1, 32).sum(-1)
        assert torch.allclose(tf.attn[name], ref, atol=1e-5), name


def test_chunk_rows_sum_to_heads():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    H = t.config.num_attention_heads
    assert torch.allclose(tf.attn["last"].sum(-1), torch.full((ids.shape[0],), float(H)), atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("W,n", [(10**6, 16), (40, 0)])
def test_no_far_context_equals_window_exactly(dtype, W, n):
    """No far context (window covers everything, or n = 0 far chunks) -> S1 equals the plain window bit for bit."""
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    rw, lw = _drafts(d, t, ids, tf, Window({"W": W, "keep": "prompt"}))
    rs, ls = _drafts(d, t, ids, tf, FarSel({"W": W, "keep": "prompt", "n": n, "score": "last"}))
    key = lambda rs_: [(r["t"], r["a"], r["tau"], r["ctx_len"]) for r in rs_]  # noqa: E731
    assert key(rw) == key(rs)
    assert len(lw) == len(ls) > 10 and all(torch.equal(a, b) for a, b in zip(lw, ls))


@pytest.mark.parametrize("score", ["last", "mid3", "random"])
def test_far_selection_equals_fresh_forward(score):
    """Active far chunks: z = plain drafter forward over prompt + chosen chunks + window, contiguous positions; the
    chosen chunks are the top-n of the target row at start-1 (or the seeded random draw)."""
    W, n, C = 40, 2, 32
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    zs, picks = [], []

    class Spy(FarSel):
        def apply(self, z, slots):
            zs.append(z.clone())
            return z

    orig_sel = rp.WindowCache._select_far

    def sel(self, start):
        orig_sel(self, start)
        picks.append(list(self.far_chunks))

    rp.WindowCache._select_far = sel
    try:
        rounds = simulate(d, t, ids, P, tf, Spy({"W": W, "keep": "prompt", "n": n, "score": score, "seed": 3}), NEW)
    finally:
        rp.WindowCache._select_far = orig_sel
    scale = float(_draft_value(d.config, "input_embedding_scale", 1.0))
    checked = 0
    for r, z, pk in zip(rounds, zs, picks):
        s = r["t"]
        lo, hi = -(-P // C), (s - W) // C
        if hi - lo < 1:
            assert pk == []
            continue
        if s + 16 > ids.shape[0]:  # short final block: reference forward below assumes 16 slots
            continue
        assert len(pk) == min(n, hi - lo) and r["far"] == len(pk) and pk == sorted(pk)
        if score != "random":
            row = tf.attn[score][s - 1, lo:hi]
            assert set(pk) == set((row.topk(len(pk)).indices + lo).tolist())
        rows = torch.cat([torch.arange(0, P)] + [torch.arange(c * C, c * C + C) for c in pk] + [torch.arange(s - W, s)])
        assert r["ctx_len"] == rows.numel()
        block = torch.full((1, 16), d.mask_token_id, dtype=torch.long)
        block[0, 0] = ids[s]
        with torch.no_grad():
            ref = d(target_hidden=tf.ctx[rows][None], noise_embedding=_raw_input_embeddings(t, block, scale),
                    position_ids=torch.arange(0, rows.numel() + 16)[None], past_key_values=_make_cache(d.config),
                    use_cache=True)[:, -15:, :][0]
        assert torch.allclose(z, ref, atol=1e-5, rtol=1e-5), (s, (z - ref).abs().max())
        checked += 1
    assert checked > 20


def test_guard_allows_only_rows_up_to_last_committed(monkeypatch):
    """The selector reads only row start-1 of the target attention summary, always below the frontier; a planted
    read of row start (the next round's anchor) raises."""
    from rlsdk.guard import LeakageError
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    reads = []

    class Rec(Guarded):
        def __getitem__(self, idx):
            if self.name.startswith("attn"):
                reads.append((idx, self._f.t))
            return super().__getitem__(idx)

    monkeypatch.setattr(rp, "Guarded", Rec)
    simulate(d, t, ids, P, tf, FarSel({"W": 40, "keep": "prompt", "n": 2, "score": "last"}), NEW)
    assert reads and all(isinstance(i, int) and i == f - 1 for i, f in reads)
    g = Rec(tf.attn["last"], rp.Frontier(100), "attn_last")
    g[99]
    with pytest.raises(LeakageError):
        g[100]


def test_random_far_chunks_deterministic_and_different_from_guided():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = _tf(t, d, ids)
    cfg = {"W": 40, "keep": "prompt", "n": 2, "seed": 0}
    a, la = _drafts(d, t, ids, tf, FarSel({**cfg, "score": "random"}))
    b, lb = _drafts(d, t, ids, tf, FarSel({**cfg, "score": "random"}))
    g, lg = _drafts(d, t, ids, tf, FarSel({**cfg, "score": "last"}))
    assert all(torch.equal(x, y) for x, y in zip(la, lb)) and len(la) == len(lb)
    assert not (len(la) == len(lg) and all(torch.equal(x, y) for x, y in zip(la, lg)))
