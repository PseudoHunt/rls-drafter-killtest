"""TTS with window / S2 drafter context, stride and pooled steps (P3 arms) on the tiny model (CPU)."""
import pytest
import torch

import rlsdk.adapters as ads
from rlsdk.adapters import TTS
from rlsdk.guard import LeakageError
from rlsdk.replay import simulate, teacher_forced
from rlsdk.target_attn import default_spec
from test_replay import NEW, P, _generate, _setup

SMALL = {"window": {"W": 40, "keep": "prompt", "positions": "contiguous"},
         "s2": {"W": 40, "keep": "prompt", "positions": "contiguous", "far": {"n": 2, "score": "mid3", "seed": 0}}}


def _trace():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    return t, d, ids, teacher_forced(t, ids, d.target_layer_ids, attn_spec=default_spec(t))


def _run(t, d, ids, tf, cfg, cls=TTS):
    pristine = {k: v.clone() for k, v in d.state_dict().items()}
    ad = cls(cfg)
    ad.bind(d, pristine)
    rounds = simulate(d, t, ids, P, tf, ad, NEW)
    masters = [m.clone() for m in ad.masters]
    diag = ad.diagnostics()
    ad.finish()
    assert all(torch.equal(pristine[k], v) for k, v in d.state_dict().items())
    return rounds, masters, diag


@pytest.mark.parametrize("ctx", ["window", "s2"])
def test_ctx_covering_whole_trace_equals_full_context_tts(ctx):
    """Frozen window / S2 settings (W 1024 > tiny traces): no context is dropped -> identical to plain TTS."""
    t, d, ids, tf = _trace()
    r0, m0, _ = _run(t, d, ids, tf, {"lr": 1e-2})
    r1, m1, _ = _run(t, d, ids, tf, {"lr": 1e-2, "ctx": ctx})
    key = lambda rs: [(r["t"], r["a"], r["tau"]) for r in rs]  # noqa: E731
    assert key(r0) == key(r1)
    assert all(torch.equal(a, b) for a, b in zip(m0, m1))


@pytest.mark.parametrize("ctx", ["window", "s2"])
def test_grad_forward_attends_to_drafting_keys(ctx, monkeypatch):
    """Active window / far chunks: the gradient forward reproduces the round's drafting z (same visible keys and
    positions, parameters unchanged since drafting), i.e. the backward sees only the kept keys."""
    monkeypatch.setitem(ads.TTS_CTX, ctx, SMALL[ctx])
    t, d, ids, tf = _trace()
    checked = []

    class Spy(TTS):
        def update(self, z, h, slots, rnd):
            with torch.no_grad():
                zg = rnd.grad_forward()[:z.shape[0]]
            assert torch.allclose(zg, z, atol=1e-5, rtol=1e-5), (zg - z).abs().max()
            checked.append(rnd.t)
            return super().update(z, h, slots, rnd)

    rounds, _, _ = _run(t, d, ids, tf, {"lr": 1e-2, "ctx": ctx}, Spy)
    assert len(checked) > 50
    if ctx == "s2":
        assert sum(r.get("far", 0) > 0 for r in rounds) > 20
    assert max(r["ctx_len"] for r in rounds) < ids.shape[0] - 100  # context really windowed


@pytest.mark.parametrize("pool", [False, True])
def test_stride_schedule(pool):
    t, d, ids, tf = _trace()
    rounds, _, diag = _run(t, d, ids, tf, {"lr": 1e-2, "stride": 10, "pool": pool})
    upd = [r for r in rounds if r["m"] > 0]
    assert diag["n_updates"] == len(upd) and diag["n_steps"] == len(upd) // 10
    for i, r in enumerate(upd, 1):
        assert r["step"] == int(i % 10 == 0)
        assert r["grad"] == (1 if pool else int(i % 10 == 0))


def test_stride_changes_trajectory_and_stride1_pool_is_plain():
    t, d, ids, tf = _trace()
    key = lambda rs: [(r["t"], r["a"]) for r in rs]  # noqa: E731
    r1, m1, _ = _run(t, d, ids, tf, {"lr": 1e-2})
    rp, mp, _ = _run(t, d, ids, tf, {"lr": 1e-2, "pool": True})
    assert key(r1) == key(rp) and all(torch.equal(a, b) for a, b in zip(m1, mp))
    r10, _, _ = _run(t, d, ids, tf, {"lr": 1e-2, "stride": 10})
    assert key(r10) != key(r1)


def test_pooled_step_uses_sum_of_rounds_gradients():
    """At each pooled step the master gradient equals the sum of the k rounds' individual gradients (each computed
    under its own round's context, parameters fixed within the k rounds)."""
    t, d, ids, tf = _trace()
    per_round, at_step = [], []

    class Spy(TTS):
        def _grad(self, m, rnd):
            before = [None if mp.grad is None else mp.grad.clone() for mp in self.masters]
            out = super()._grad(m, rnd)
            per_round.append([mp.grad - (0 if b is None else b) for mp, b in zip(self.masters, before)])
            return out

        def _step(self):
            at_step.append(([mp.grad.clone() for mp in self.masters], len(per_round)))
            return super()._step()

    ads.TTS_CTX["_test"] = SMALL["s2"]
    try:
        _run(t, d, ids, tf, {"lr": 1e-2, "stride": 4, "pool": True, "ctx": "_test"}, Spy)
    finally:
        del ads.TTS_CTX["_test"]
    assert len(at_step) > 10
    for grads, n in at_step:
        rows = per_round[n - 4:n]
        for j, g in enumerate(grads):
            assert torch.allclose(g, sum(r[j] for r in rows), atol=1e-6)


@pytest.mark.parametrize("ctx", ["window", "s2"])
def test_guard_still_raises_with_ctx_tts(ctx, monkeypatch):
    monkeypatch.setitem(ads.TTS_CTX, ctx, SMALL[ctx])
    t, d, ids, tf = _trace()
    pristine = {k: v.clone() for k, v in d.state_dict().items()}
    ad = TTS({"lr": 1e-2, "ctx": ctx, "stride": 10, "pool": True})
    ad.bind(d, pristine)
    seen = []

    def hook(rnd):
        seen.append(rnd.t)
        if len(seen) == 30:
            rnd.hstar[rnd.frontier.t]

    with pytest.raises(LeakageError):
        simulate(d, t, ids, P, tf, ad, NEW, round_hook=hook)
    ad.finish()
