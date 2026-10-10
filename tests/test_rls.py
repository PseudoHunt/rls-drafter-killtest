"""Test 1: batched RLS == closed-form weighted ridge (rel err < 1e-4); plus bias and per-slot checks."""
import pytest
import torch

from rlsdk.adapters import RLS, Bias, SLOT_GROUPS, _RLSState


def _rounds(d, n_rounds, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g) / d**0.5
    out = []
    for _ in range(n_rounds):
        m = int(torch.randint(1, 17, (1,), generator=g))
        z = torch.randn(m, d, generator=g)
        h = z @ (torch.eye(d) + 0.3 * A).T + 0.1 * torch.randn(m, d, generator=g) + 0.5
        out.append((z, h))
    return out


def _closed_form(rounds, d, delta, lam, prior):
    """argmin_M sum_j w_j ||h_j - M z~_j||^2 + delta_eff ||M - M0||^2, with the weights implied by
    per-token forgetting lambda_r = lambda^m (token weight = lambda^(tokens in later rounds))."""
    D = d + 1
    total = sum(z.shape[0] for z, _ in rounds)
    M0 = torch.eye(d, D, dtype=torch.float64)
    A = torch.zeros(D, D, dtype=torch.float64)
    B = torch.zeros(d, D, dtype=torch.float64)
    later = total
    for z, h in rounds:
        later -= z.shape[0]
        w = lam**later
        zt = torch.cat([z, torch.ones(z.shape[0], 1)], 1).double()
        A += w * zt.T @ zt
        B += w * h.double().T @ zt
    de = delta * (lam**total if prior == "decay" else 1.0)
    return (B + de * M0) @ torch.linalg.inv(A + de * torch.eye(D, dtype=torch.float64))


@pytest.mark.parametrize("prior", ["decay", "fixed"])
@pytest.mark.parametrize("lam", [0.98, 0.999, 1.0])
def test_rls_matches_weighted_ridge(prior, lam):
    d, delta = 24, 3.0
    rounds = _rounds(d, 40, seed=1)
    st = _RLSState(d, delta, lam, prior, "cpu")
    for z, h in rounds:
        st.update(z, h)
    M = st.M().double()
    M_ref = _closed_form(rounds, d, delta, lam, prior)
    rel = (M - M_ref).norm() / M_ref.norm()
    assert rel < 1e-4, float(rel)


@pytest.mark.parametrize("prior", ["decay", "fixed"])
def test_rls_apply_equals_M_ztilde(prior):
    d = 12
    rounds = _rounds(d, 10, seed=2)
    ad = RLS({"lam": 0.99, "n0": 4, "sigma2": 1.0, "prior": prior})
    ad.reset(d, "cpu")
    for z, h in rounds:
        ad.update(z, h, torch.arange(1, z.shape[0] + 1), None)
    z = torch.randn(5, d)
    M = ad.groups[0][2].M()
    ref = torch.cat([z, torch.ones(5, 1)], 1) @ M.T
    assert torch.allclose(ad.apply(z, torch.arange(1, 6)), ref, atol=1e-4)


def test_per_slot_rls_routes_groups():
    d = 8
    ad = RLS({"lam": 0.99, "n0": 4, "sigma2": 1.0, "per_slot": True})
    ad.reset(d, "cpu")
    z, h = torch.randn(15, d), torch.randn(15, d)
    ad.update(z, h, torch.arange(1, 16), None)
    sizes = [st.updates for _, _, st in ad.groups]
    assert sizes == [1, 1, 1] and [g[:2] for g in ad.groups] == list(SLOT_GROUPS)
    # group {1} state only saw slot-1 data: equals a fresh single state updated with row 0
    ref = _RLSState(d, ad.delta, 0.99, "decay", "cpu")
    ref.update(z[:1], h[:1])
    assert torch.allclose(ad.groups[0][2].dM, ref.dM)


def test_bias_is_exponentially_weighted_mean():
    d, rho = 6, 0.9
    ad = Bias({"rho": rho})
    ad.reset(d, "cpu")
    res = []
    for m in (3, 1, 5):
        z, h = torch.randn(m, d), torch.randn(m, d)
        ad.update(z, h, None, None)
        res.extend(list(h - z))
    n = len(res)
    w = torch.tensor([rho ** (n - 1 - j) for j in range(n)])
    ref = (w[:, None] * torch.stack(res)).sum(0) / w.sum()
    z = torch.randn(2, d)
    assert torch.allclose(ad.apply(z, None) - z, ref.expand(2, d), atol=1e-5)
