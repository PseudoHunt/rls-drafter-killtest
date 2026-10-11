"""Exact batched pooled TTS update (rlsdk/batched.py) on the tiny model (CPU)."""
import pytest
import torch

import rlsdk.adapters as ads
import rlsdk.batched as bt
from test_tts_ctx import SMALL, _run, _trace


@pytest.fixture
def s2_small(monkeypatch):
    monkeypatch.setitem(ads.TTS_CTX, "_test", SMALL["s2"])
    return "_test"


@pytest.mark.parametrize("stride", [4, 10])
def test_batched_gradient_equals_per_round_accumulated(stride, s2_small):
    """At every step the batched gradient equals the per-round accumulated gradient of the same k rounds (same
    trajectory; each round under its own S2 visible keys and stale context K/V): rel. error < 1e-3."""
    t, d, ids, tf = _trace()
    rounds, _, diag = _run(t, d, ids, tf, {"lr": 1e-2, "stride": stride, "pool": True, "batched": True,
                                           "check": True, "ctx": s2_small})
    errs = diag["check_rel_err"]
    assert len(errs) == diag["n_steps"] > 10
    assert max(errs) < 1e-3, errs
    assert sum(r.get("far", 0) > 0 for r in rounds) > 20  # far chunks active
    assert max(r["ctx_len"] for r in rounds) < ids.shape[0] - 100  # window active


def test_buffers_append_only(s2_small, monkeypatch):
    """Stale-cache semantics: context K/V rows are never rewritten after they are first fed, so the indices stored
    at round time read, at step time, exactly the K/V the round attended to."""
    t, d, ids, tf = _trace()
    snaps = []
    orig = bt.batched_loss

    def spy(drafter, head, recs, weights):
        c = recs[0].cache
        lo = min(r.cache.c0 for r in recs)  # rows < lo were fed before the first pooled round
        snaps.append((lo, [k[:, :, :lo].clone() for k in c.k], [v[:, :, :lo].clone() for v in c.v], c))
        return orig(drafter, head, recs, weights)

    monkeypatch.setattr(bt, "batched_loss", spy)
    _run(t, d, ids, tf, {"lr": 1e-2, "stride": 4, "pool": True, "batched": True, "ctx": s2_small})
    assert len(snaps) > 10
    c = snaps[0][3]
    for lo, ks, vs, _ in snaps:  # every earlier snapshot still equals the final buffers
        assert all(torch.equal(k, ck[:, :, :lo]) for k, ck in zip(ks, c.k))
        assert all(torch.equal(v, cv[:, :, :lo]) for v, cv in zip(vs, c.v))


def test_rebuilt_kv_would_fail_the_check(s2_small, monkeypatch):
    """Negative control: recomputing the older context K/V with the current weights at step time (instead of the
    stale K/V the rounds saw) breaks gradient equality after the first step -> the equality test discriminates."""
    t, d, ids, tf = _trace()
    orig = bt.batched_loss

    def rebuilt(drafter, head, recs, weights):
        c = recs[0].cache
        hi = max(r.cache.start for r in recs)
        with torch.no_grad():
            th = drafter.hidden_norm(drafter.fc(tf.ctx[:hi][None]))
            saved = [(k.clone(), v.clone()) for k, v in zip(c.k, c.v)]
            for li, layer in enumerate(drafter.layers):
                a = layer.self_attn
                c.k[li][:, :, :hi] = a.k_norm(a.k_proj(th).view(1, hi, -1, a.head_dim)).transpose(1, 2)
                c.v[li][:, :, :hi] = a.v_proj(th).view(1, hi, -1, a.head_dim).transpose(1, 2)
        try:
            return orig(drafter, head, recs, weights)
        finally:
            for li, (k, v) in enumerate(saved):
                c.k[li].copy_(k)
                c.v[li].copy_(v)

    monkeypatch.setattr(bt, "batched_loss", rebuilt)
    _, _, diag = _run(t, d, ids, tf, {"lr": 1e-2, "stride": 4, "pool": True, "batched": True, "check": True,
                                      "ctx": s2_small})
    errs = diag["check_rel_err"]
    assert errs[0] < 1e-3  # before any step, rebuilt == stale
    assert max(errs[1:]) > 1e-2


def test_batched_schedule_and_trajectory_match_pooled(s2_small):
    t, d, ids, tf = _trace()
    cfg = {"lr": 1e-2, "stride": 10, "pool": True, "ctx": s2_small}
    rp, mp, _ = _run(t, d, ids, tf, cfg)
    rb, mb, diag = _run(t, d, ids, tf, {**cfg, "batched": True})
    upd = [r for r in rb if r["m"] > 0]
    assert diag["n_steps"] == len(upd) // 10
    for i, r in enumerate(upd, 1):
        assert r["step"] == r["grad"] == int(i % 10 == 0)
    key = lambda rs: [(r["t"], r["a"]) for r in rs]  # noqa: E731
    assert key(rp) == key(rb)
    p0 = [p.detach().float() for p in d.parameters()]  # pristine (restored by finish)
    num = sum(float(((a - b) ** 2).sum()) for a, b in zip(mp, mb))
    den = sum(float(((a - p) ** 2).sum()) for a, p in zip(mp, p0))
    assert den > 0 and (num / den) ** 0.5 < 1e-3  # total weight change agrees


def test_batched_requires_pool_and_ctx():
    with pytest.raises(ValueError):
        ads.TTS({"lr": 1e-5, "stride": 10, "batched": True, "ctx": "s2"})
    with pytest.raises(ValueError):
        ads.TTS({"lr": 1e-5, "stride": 10, "pool": True, "batched": True})


def test_warm_schedule_then_batched(s2_small):
    """warm: per-round steps while the round starts before output position `warm`, then batched pooled stride k
    (counted from the first post-warm round), with the batched gradient still exact."""
    from test_replay import P
    t, d, ids, tf = _trace()
    warm = 120
    rounds, _, diag = _run(t, d, ids, tf, {"lr": 1e-2, "stride": 4, "pool": True, "batched": True, "check": True,
                                           "warm": warm, "ctx": s2_small})
    upd = [r for r in rounds if r["m"] > 0]
    pre = [r for r in upd if r["pos"] < warm]
    post = [r for r in upd if r["pos"] >= warm]
    assert pre and post and all(r["step"] == r["grad"] == 1 for r in pre)
    for i, r in enumerate(post, 1):
        assert r["step"] == r["grad"] == int(i % 4 == 0)
    assert diag["n_steps"] == len(pre) + len(post) // 4
    assert len(diag["check_rel_err"]) == len(post) // 4 and max(diag["check_rel_err"]) < 1e-3
    assert P > 0
