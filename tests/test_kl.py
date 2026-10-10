"""KL arms (KL-bias, KL-linear) on the tiny model (CPU)."""
import pytest
import torch

from rlsdk.adapters import Baseline, KLBias, KLLinear, dflash_position_weights
from rlsdk.guard import LeakageError
from rlsdk.replay import simulate, teacher_forced
from test_replay import NEW, P, _generate, _setup

KL_ARMS = [lambda: KLBias({"rho": 0.9, "alpha": 0.5}), lambda: KLLinear({"lr": 1e-2})]


def _trace():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    return t, d, ids, teacher_forced(t, ids, d.target_layer_ids)


def _kl_loss(W, zp, h, w):
    p_logp = torch.log_softmax(h @ W.T, -1)
    q_logp = torch.log_softmax(zp @ W.T, -1)
    return (w * (p_logp.exp() * (p_logp - q_logp)).sum(-1)).sum()


def _inputs(m=5, d=32, V=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(V, d, generator=g, dtype=torch.float64)
    z = torch.randn(m, d, generator=g, dtype=torch.float64)
    h = torch.randn(m, d, generator=g, dtype=torch.float64)
    slots = torch.arange(1, m + 1)
    return W, z, h, slots


def test_kllinear_closed_form_gradient_equals_autograd():
    W, z, h, slots = _inputs()
    ad = KLLinear({"lr": 1e-3})
    ad.W = W
    ad.reset(32, "cpu")
    ad.weights = ad.weights.double()
    A = (0.05 * torch.randn(32, 33, generator=torch.Generator().manual_seed(1), dtype=torch.float64)).requires_grad_()
    zt = ad._tilde(z)
    zp = z + zt @ A.T
    _kl_loss(W, zp, h, dflash_position_weights(5).double()).backward()
    closed = ad.grad(z, h, slots, zp.detach())
    assert torch.allclose(closed, A.grad, rtol=1e-10, atol=1e-12)


def test_klbias_direction_is_negative_autograd_gradient():
    """g = sum_i w_i W^T (p_i - q_i) = -d/db sum_i w_i KL(p_i || softmax(W (z_i + b))) at the drafting z."""
    W, z, h, slots = _inputs()
    ad = KLBias({"rho": 0.9, "alpha": 0.1})
    ad.W = W
    ad.reset(32, "cpu")
    ad.weights = ad.weights.double()
    b = torch.zeros(32, dtype=torch.float64, requires_grad=True)
    _kl_loss(W, z + b, h, dflash_position_weights(5).double()).backward()
    g = (ad._p_minus_q(z, h, slots) @ W).sum(0)
    assert torch.allclose(g, -b.grad, rtol=1e-10, atol=1e-12)


def test_klbias_momentum_and_apply():
    W, z, h, slots = _inputs()
    W, z, h = W.float(), z.float(), h.float()
    ad = KLBias({"rho": 0.9, "alpha": 0.1})
    ad.W = W
    ad.reset(32, "cpu")
    assert torch.equal(ad.apply(z, slots), z)  # no update yet
    ad.update(z, h, slots, None)
    g = (ad._p_minus_q(z, h, slots) @ W).sum(0)
    mom = 0.1 * g / g.pow(2).mean().sqrt()
    assert torch.allclose(ad.mom, mom)
    zp = ad.apply(z, slots)
    ref = z + 0.1 * z.pow(2).mean(-1, keepdim=True).sqrt() * mom / mom.pow(2).mean().sqrt()
    assert torch.allclose(zp, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("make", [lambda: KLBias({"rho": 0.9, "alpha": 0.0}), lambda: KLLinear({"lr": 0.0})])
def test_zero_adaptation_reproduces_baseline_bit_exact(dtype, make):
    """alpha = 0 (KL-bias) and A = 0 (KL-linear, lr = 0 keeps A = 0): every round's draft logits and the round
    log equal the baseline bit for bit, over the whole trace."""
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    orig = d.compute_logits
    logs = {}
    for name, ad in (("base", Baseline()), ("kl", make())):
        seen = logs.setdefault(name, [])
        d.compute_logits = lambda hh, head, seen=seen: seen.append(orig(hh, head)) or seen[-1]
        rounds = simulate(d, t, ids, P, tf, ad, NEW)
        logs[name + "_rounds"] = [(r["t"], r["a"], r["tau"]) for r in rounds]
    d.compute_logits = orig
    assert len(logs["base"]) == len(logs["kl"]) > 10
    assert all(torch.equal(a, b) for a, b in zip(logs["base"], logs["kl"]))
    assert logs["base_rounds"] == logs["kl_rounds"]


@pytest.mark.parametrize("make", KL_ARMS)
def test_kl_arms_change_drafts(make):
    t, d, ids, tf = _trace()
    base = [r["tau"] for r in simulate(d, t, ids, P, tf, Baseline(), NEW)]
    rounds = simulate(d, t, ids, P, tf, make(), NEW)
    assert [r["tau"] for r in rounds] != base


@pytest.mark.parametrize("make", KL_ARMS)
def test_guard_still_raises_with_kl_arms(make):
    t, d, ids, tf = _trace()
    seen = []

    def hook(rnd):
        rnd.hstar[rnd.t:rnd.t + rnd.m]
        seen.append(rnd.t)
        if len(seen) == 3:
            rnd.hstar[rnd.frontier.t]  # planted leak

    with pytest.raises(LeakageError):
        simulate(d, t, ids, P, tf, make(), NEW, round_hook=hook)
    assert len(seen) == 3


@pytest.mark.parametrize("make", KL_ARMS)
def test_only_valid_pairs_used(make):
    """Updates see exactly slots 1..a+1 (capped at the block / trace end) with h* rows t..t+a, and the update
    is unaffected by what the drafter produced at slots > a+1."""
    t, d, ids, tf = _trace()
    ad = make()
    calls = []
    orig_update = ad.update

    def update(z, h, slots, rnd):
        calls.append((rnd.t, slots.tolist(), torch.equal(h, tf.hstar[rnd.t:rnd.t + rnd.m])))
        if isinstance(ad, KLBias):
            ad._zp = torch.cat([ad._zp[:z.shape[0]], torch.full_like(ad._zp[z.shape[0]:], 1e6)])  # poison > a+1
        else:
            ad._zp = torch.cat([ad._zp[:z.shape[0]], torch.full_like(ad._zp[z.shape[0]:], float("nan"))])
        return orig_update(z, h, slots, rnd)

    ad.update = update
    rounds = simulate(d, t, ids, P, tf, ad, NEW)
    by_t = {r["t"]: r for r in rounds}
    assert len(calls) == sum(r["m"] > 0 for r in rounds)
    for tt, slots, h_ok in calls:
        r = by_t[tt]
        assert slots == list(range(1, min(r["a"] + 1, 15, r["tau"]) + 1)) and h_ok
    assert all(torch.isfinite(x).all() for x in ([ad.mom] if isinstance(ad, KLBias) else [ad.A]))
    assert any(r["a"] < 14 for r in rounds)


def test_kllinear_logs_relative_correction():
    t, d, ids, tf = _trace()
    ad = KLLinear({"lr": 1e-2})
    rounds = simulate(d, t, ids, P, tf, ad, NEW)
    assert all("rel" in r for r in rounds)
    assert rounds[0]["rel"] == 0.0 and max(r["rel"] for r in rounds) > 0
    diag = ad.diagnostics()
    assert diag["n_rounds_rel_gt_flag"] == sum(r["rel"] > 0.2 for r in rounds)
    assert ad.A.dtype == torch.float32 and ad.opt.state[ad.A]["exp_avg"].dtype == torch.float32
