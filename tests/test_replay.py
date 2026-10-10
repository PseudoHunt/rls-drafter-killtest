"""Tests 2-4 and harness checks on the tiny model (CPU)."""
import pytest
import torch
from dflash.model import _make_cache, dflash_generate, extract_context_feature

from rlsdk.adapters import RLS, TTS, Baseline, Bias
from rlsdk.guard import Frontier, Guarded, LeakageError
from rlsdk.replay import rebuild_context_cache, simulate, teacher_forced
from rlsdk.tiny import tiny_drafter, tiny_target

P, NEW = 24, 300


def _setup(dtype=torch.float32, seed=0):
    t = tiny_target(dtype=dtype)
    d = tiny_drafter(t, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, 7, (1, P), generator=g)
    return t, d, prompt


def _generate(t, d, prompt, new=NEW, stop=None):
    return dflash_generate(d, t, prompt, new, stop, return_stats=True)


def test_hook_features_match_output_hidden_states():
    t, d, prompt = _setup()
    out = _generate(t, d, prompt)
    ids = out.output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    with torch.no_grad():
        ref = t(ids[None], output_hidden_states=True)
    ctx_ref = extract_context_feature(ref.hidden_states, d.target_layer_ids)[0]
    assert torch.allclose(tf.ctx, ctx_ref, atol=1e-6)
    assert torch.equal(tf.argmax, ref.logits[0].argmax(-1))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("stop", [None, [3]])
def test_replay_equals_real_speculative_loop(seed, stop):
    """Test 4: per-round committed counts from replay == dflash_generate's acceptance_lengths."""
    t, d, prompt = _setup(seed=seed)
    out = _generate(t, d, prompt, stop=stop)
    ids = out.output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    rounds = simulate(d, t, ids, P, tf, Baseline(), NEW)
    assert [r["tau"] for r in rounds] == out.acceptance_lengths
    assert max(out.acceptance_lengths) > 1  # non-trivial acceptance exercised


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prior", ["decay", "fixed"])
def test_rls_at_M0_reproduces_baseline_bit_exact(dtype, prior):
    """Test 2: RLS at M0 gives bit-identical first-round drafts (and logits) to the baseline."""
    t, d, prompt = _setup(dtype=dtype)
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    captured = {}
    import rlsdk.replay as rp
    orig = d.compute_logits

    def spy(name):
        def f(h, head):
            captured.setdefault(name, orig(h, head))
            return captured[name]
        return f

    for name, ad in (("base", Baseline()), ("rls", RLS({"lam": 0.99, "n0": 16, "sigma2": 1.0, "prior": prior}))):
        d.compute_logits = spy(name)
        simulate(d, t, ids, P, tf, ad, NEW)
    d.compute_logits = orig
    assert torch.equal(captured["base"], captured["rls"])


def test_guard_raises_on_planted_read():
    """Test 3a: direct reads at/after the frontier raise; below it they succeed."""
    f = Frontier(10)
    g = Guarded(torch.arange(20), f)
    assert g[0:10].tolist() == list(range(10))
    assert int(g[9]) == 9
    for bad in (lambda: g[10], lambda: g[5:11], lambda: g[torch.tensor([3, 12])], lambda: g[-1], lambda: g[10:]):
        with pytest.raises(LeakageError):
            bad()
    f.advance(15)
    assert int(g[14]) == 14
    with pytest.raises(LeakageError):
        g[15]


def test_guard_in_replay_blocks_lookahead():
    """Test 3b: inside the replay, a read of h* at the new frontier (first unrevealed position) raises,
    while the valid supervision rows are readable."""
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    seen = []

    def hook(rnd):
        rnd.hstar[rnd.t:rnd.t + rnd.m]  # allowed
        assert rnd.frontier.t == rnd.t + rnd.m or rnd.m == 15
        seen.append(rnd.t)
        rnd.hstar[rnd.frontier.t]  # planted leak

    with pytest.raises(LeakageError):
        simulate(d, t, ids, P, tf, Baseline(), NEW, round_hook=hook)
    assert len(seen) == 1


def test_rebuild_cache_matches_incremental():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    cache = _make_cache(d.config)
    with torch.no_grad():
        # build incrementally in irregular chunks the way rounds do
        cuts = [0, P, P + 3, P + 4, P + 11, P + 30]
        for a, b in zip(cuts[:-1], cuts[1:]):
            noise = torch.zeros(1, 16, d.config.hidden_size)
            pos = torch.arange(a, b + 16)[None]
            d(target_hidden=tf.ctx[a:b][None], noise_embedding=noise, position_ids=pos,
              past_key_values=cache, use_cache=True)
            cache.crop(-(cache.get_seq_length() - b))
        inc = [(l.keys.clone(), l.values.clone()) for l in cache.layers]
        rebuild_context_cache(d, cache, tf.ctx[:cuts[-1]])
    for (k0, v0), l in zip(inc, cache.layers):
        assert torch.allclose(k0, l.keys, atol=1e-5) and torch.allclose(v0, l.values, atol=1e-5)


@pytest.mark.parametrize("refresh", [False, True])
def test_tts_runs_and_changes_drafts(refresh):
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    pristine = {k: v.clone() for k, v in d.state_dict().items()}
    ad = TTS({"lr": 1e-2, "refresh": refresh})
    ad.bind(d, pristine)
    rounds = simulate(d, t, ids, P, tf, ad, NEW)
    changed = any(not torch.equal(pristine[k], v) for k, v in d.state_dict().items())
    ad.finish()
    assert changed
    assert all(torch.equal(pristine[k], v) for k, v in d.state_dict().items())
    base = simulate(d, t, ids, P, tf, Baseline(), NEW)
    assert [r["tau"] for r in rounds] != [r["tau"] for r in base]
    assert all(r["update_ms"] > 0 for r in rounds if r["m"] > 0)


def test_bias_and_rls_change_drafts():
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    base = [r["tau"] for r in simulate(d, t, ids, P, tf, Baseline(), NEW)]
    for ad in (Bias({"rho": 0.9}), RLS({"lam": 0.99, "n0": 1, "sigma2": 0.1})):
        assert [r["tau"] for r in simulate(d, t, ids, P, tf, ad, NEW)] != base


def test_rls_checks_lockstep_tiny():
    """scripts/rls_checks.Lockstep on the tiny model: lam=1 recursion == fixed; fp32 ~ fp64."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "rls_checks", os.path.join(os.path.dirname(__file__), "..", "scripts", "rls_checks.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids)
    W = t.lm_head.weight.double()
    base = {"n0": 4, "sigma2": 1.0}
    lk = rc.Lockstep(RLS({**base, "lam": 1.0, "prior": "decay"}), RLS({**base, "lam": 1.0, "prior": "fixed"}),
                     W, "lam1", every=20)
    simulate(d, t, ids, P, tf, lk, NEW)
    assert lk._mrel()["rel_M"] < 1e-4
    lk = rc.Lockstep(RLS({**base, "lam": 0.99, "dtype": "fp32"}), RLS({**base, "lam": 0.99, "dtype": "fp64"}), W, "fp64")
    rounds = simulate(d, t, ids, P, tf, lk, NEW)
    assert lk.max_logit_diff < 1e-3
    assert all("solve_ms" in r for r in rounds)


@pytest.mark.parametrize("c", [37, 120, 201, 299])
@pytest.mark.parametrize("method", ["baseline", "rls"])
def test_truncation_derivation_exact(c, method):
    """Amendment 2: metrics on a trace truncated to c output tokens == apply_cut(full-trace replay, c)."""
    from rlsdk.analyze import apply_cut
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    mk = lambda: Baseline() if method == "baseline" else RLS({"lam": 0.99, "n0": 4, "sigma2": 1.0})
    full = simulate(d, t, ids, P, teacher_forced(t, ids, d.target_layer_ids), mk(), NEW)
    tids = ids[:P + c]
    trunc = simulate(d, t, tids, P, teacher_forced(t, tids, d.target_layer_ids), mk(), NEW)
    key = lambda rs: [(r["t"], r["tau"]) for r in rs]
    assert key(trunc) == key(apply_cut(full, c))
