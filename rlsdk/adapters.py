"""Test-time adaptation methods (brief §5).

All adapters see the drafter's per-slot lm_head input z (rows = slots 1..S of one round) and are
updated after verification with the valid pairs (z_i, h*_{t+i-1}), i = 1..m.  Adaptation math is
fp32; the adapted z is cast back to z.dtype so draft logits follow the same dtype path for every
method.
"""
import math
import time

import torch

SLOT_GROUPS = ((1, 1), (2, 4), (5, 15))  # §5.5 per-slot RLS groups (inclusive, 1-based slots)


class Baseline:
    method = "baseline"

    def __init__(self, config=None):
        self.config = config or {}
        self.z2_sum = 0.0  # per-dimension second moment of z, for sigma^2 (§5.3)
        self.z2_count = 0

    def reset(self, d, device):
        pass

    def apply(self, z, slots):
        zf = z.float()
        self.z2_sum += float((zf * zf).sum())
        self.z2_count += zf.numel()
        return z

    def update(self, z, h, slots, rnd):
        pass

    def state_bytes(self):
        return 0


class Bias:
    """z' = z + b, b = exponentially weighted mean of residuals (h* - z), per-token decay rho.

    b = sum_j rho^age_j r_j / sum_j rho^age_j (normalized weighted mean; age counted in tokens,
    tokens within a round ordered by slot).
    """

    method = "bias"

    def __init__(self, config):
        self.config = config
        self.rho = float(config["rho"])

    def reset(self, d, device):
        self.num = torch.zeros(d, dtype=torch.float32, device=device)
        self.den = 0.0

    def apply(self, z, slots):
        if self.den == 0.0:
            return z
        return (z.float() + self.num / self.den).to(z.dtype)

    def update(self, z, h, slots, rnd):
        m = z.shape[0]
        r = h.float() - z.float()
        w = self.rho ** torch.arange(m - 1, -1, -1, dtype=torch.float32, device=r.device)
        self.num = self.rho**m * self.num + (w[:, None] * r).sum(0)
        self.den = self.rho**m * self.den + float(w.sum())

    def state_bytes(self):
        return self.num.numel() * 4


class _RLSState:
    """One (dM, P) pair.  M = M0 + dM with M0 = [I | 0]; the update is the brief's batched RLS
    written in terms of dM (algebraically identical, keeps M0 rows exact).

    prior="decay": literal recursion, P0 = I/delta, lambda_r = lambda^m.  Equivalent to weighted ridge
        where a token's weight is lambda^(#tokens committed in later rounds) and the prior weight is
        delta * lambda^(#tokens so far).
    prior="fixed": exact minimiser of sum_tokens lambda^age ||h - M z~||^2 + delta ||M - M0||_F^2 with a
        non-decaying prior (brief §5.3 objective as written).  Keeps sufficient statistics and a
        Cholesky factor; O(d^3) per update instead of O(d^2 m).
    """

    def __init__(self, d, delta, lam, prior, device, dtype=torch.float32):
        self.d, self.delta, self.lam, self.prior = d, float(delta), float(lam), prior
        self.dtype, self.device = dtype, torch.device(device)
        self.fp64_fallback_at = None  # update index at which a failed fp32 Cholesky forced fp64
        self.solve_ms = 0.0  # accumulated linear-algebra time since last pop (Amendment 1)
        D = d + 1
        self.updates = 0
        if prior == "decay":
            self.dM = torch.zeros(d, D, dtype=dtype, device=device)
            self.P = torch.eye(D, dtype=dtype, device=device) / self.delta
        elif prior == "fixed":
            self.G = torch.zeros(D, D, dtype=dtype, device=device)
            self.B = torch.zeros(d, D, dtype=dtype, device=device)
            self.L = None
        else:
            raise ValueError(prior)

    def _now(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @staticmethod
    def _tilde(z):
        return torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)], dim=1)

    def correction(self, zf):
        """Return dM z~ for rows of zf ([n, d]), as [n, d] in zf's dtype."""
        zt = self._tilde(zf.to(self.dtype))
        if self.prior == "decay":
            return (zt @ self.dM.T).to(zf.dtype)
        if self.L is None:
            return torch.zeros_like(zf)
        t0 = self._now()
        y = torch.cholesky_solve(zt.T, self.L)  # (G + delta I)^-1 z~
        out = (self.B @ y).T.to(zf.dtype)
        self.solve_ms += (self._now() - t0) * 1e3
        return out

    def _factor(self):
        A = self.G + self.delta * torch.eye(self.d + 1, dtype=self.G.dtype, device=self.G.device)
        L, info = torch.linalg.cholesky_ex(0.5 * (A + A.T))
        if int(info) != 0 and self.dtype != torch.float64:
            self.fp64_fallback_at = self.updates
            self.dtype = torch.float64
            self.G, self.B = self.G.double(), self.B.double()
            return self._factor()
        if int(info) != 0:
            raise RuntimeError(f"Cholesky failed in fp64 at update {self.updates}")
        return L

    def update(self, zf, hf):
        m = zf.shape[0]
        lam_r = self.lam**m
        zf, hf = zf.to(self.dtype), hf.to(self.dtype)
        Z = self._tilde(zf).T  # [D, m]
        R = (hf - zf).T  # residual targets for dM: H - M0 Z  [d, m]
        t0 = self._now()
        if self.prior == "decay":
            PZ = self.P @ Z
            S = lam_r * torch.eye(m, dtype=Z.dtype, device=Z.device) + Z.T @ PZ
            K = torch.linalg.solve(S, PZ.T).T  # P Z S^-1   (S symmetric)
            E = R - self.dM @ Z
            self.dM += E @ K.T
            self.P = (self.P - K @ PZ.T) / lam_r
            self.updates += 1
            if self.updates % 64 == 0:
                self.P = 0.5 * (self.P + self.P.T)
        else:
            self.G = lam_r * self.G + Z @ Z.T
            self.B = lam_r * self.B + R @ Z.T
            self.L = self._factor()
            self.updates += 1
        self.solve_ms += (self._now() - t0) * 1e3

    def M(self, dtype=torch.float32):
        """Full M on CPU (tests / checks)."""
        M0 = torch.eye(self.d, self.d + 1, dtype=dtype)
        if self.prior == "decay":
            return M0 + self.dM.cpu().to(dtype)
        if self.L is None:
            return M0
        A_inv = torch.cholesky_inverse(self.L)
        return M0 + (self.B @ A_inv).cpu().to(dtype)

    def state_bytes(self):
        ts = [self.dM, self.P] if self.prior == "decay" else [self.G, self.B] + ([self.L] if self.L is not None else [])
        return sum(t.numel() * t.element_size() for t in ts)


class RLS:
    """z' = M z~ (§5.3); per_slot=True gives §5.5 (separate state per slot group)."""

    method = "rls"

    def __init__(self, config):
        self.config = config
        self.lam = float(config["lam"])
        self.delta = float(config["n0"]) * float(config["sigma2"])
        # Amendment 1: fixed-prior exact solver; at lam = 1 the O(d^2 m) recursion solves the same objective
        self.prior = config.get("prior", "decay" if self.lam == 1.0 else "fixed")
        self.dtype = {"fp32": torch.float32, "fp64": torch.float64}[config.get("dtype", "fp32")]
        self.per_slot = bool(config.get("per_slot", False))
        if self.per_slot:
            self.method = "rls_slot"

    def reset(self, d, device):
        groups = SLOT_GROUPS if self.per_slot else ((1, 10**9),)
        self.groups = [(lo, hi, _RLSState(d, self.delta, self.lam, self.prior, device, self.dtype))
                       for lo, hi in groups]

    def _split(self, slots):
        for lo, hi, st in self.groups:
            idx = ((slots >= lo) & (slots <= hi)).nonzero(as_tuple=True)[0]
            if idx.numel():
                yield idx, st

    def apply(self, z, slots):
        zf = z.float()
        out = zf.clone()
        for idx, st in self._split(slots):
            out[idx] = zf[idx] + st.correction(zf[idx])
        return out.to(z.dtype)

    def update(self, z, h, slots, rnd):
        zf, hf = z.float(), h.float()
        for idx, st in self._split(slots):
            st.update(zf[idx], hf[idx])

    def pop_solve_ms(self):
        ms = sum(st.solve_ms for _, _, st in self.groups)
        for _, _, st in self.groups:
            st.solve_ms = 0.0
        return ms

    def diagnostics(self):
        return {"prior": self.prior, "fp64_fallback_at": [st.fp64_fallback_at for _, _, st in self.groups]}

    def state_bytes(self):
        return sum(st.state_bytes() for _, _, st in self.groups)


def dflash_position_weights(n, gamma=7.0):
    """DFlash training loss decay (paper eq. 4): w_k = exp(-(k-1)/gamma), gamma=7 for block 16."""
    return torch.tensor([math.exp(-(k - 1) / gamma) for k in range(1, n + 1)], dtype=torch.float32)


class TTS:
    """One Adam step on all drafter parameters per round (§5.4).  fp32 master weights + Adam state;
    the bf16 drafter used for drafting is refreshed from the masters after every step.  The actual
    gradient computation needs the drafter graph, so `update` is driven by replay.py via `rnd`.
    """

    method = "tts"

    def __init__(self, config):
        self.config = config
        self.lr = float(config["lr"])
        self.refresh = bool(config.get("refresh", False))
        self.gamma = float(config.get("gamma", 7.0))

    def bind(self, drafter, pristine_state):
        self.drafter = drafter
        self.pristine = pristine_state

    def reset(self, d, device):
        self.drafter.load_state_dict(self.pristine)
        self.params = [p for p in self.drafter.parameters()]
        for p in self.params:
            p.requires_grad_(True)
        self.masters = [p.detach().float().clone() for p in self.params]
        self.opt = torch.optim.Adam(self.masters, lr=self.lr, betas=(0.9, 0.999))
        self.weights = dflash_position_weights(16, self.gamma).to(device)

    def finish(self):
        for p in self.params:
            p.requires_grad_(False)
            p.grad = None
        self.drafter.load_state_dict(self.pristine)
        self.masters = self.opt = None

    def apply(self, z, slots):
        return z

    def update(self, z, h, slots, rnd):
        """rnd.grad_forward() -> z with grad for slots 1..S; rnd.target_logits [m, V] fp32."""
        m = z.shape[0]
        with torch.enable_grad():
            zg = rnd.grad_forward()[:m]
            q_logp = torch.log_softmax(rnd.draft_logits(zg).float(), dim=-1)
            p_logp = torch.log_softmax(rnd.target_logits.float(), dim=-1)
            kl = (p_logp.exp() * (p_logp - q_logp)).sum(-1)  # KL(p || q) per slot
            loss = (self.weights[:m] * kl).sum()
            loss.backward()
        for p, mp in zip(self.params, self.masters):
            mp.grad = None if p.grad is None else p.grad.float()
            p.grad = None
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, mp in zip(self.params, self.masters):
                p.copy_(mp)
        if self.refresh:
            rnd.refresh_cache()
        return float(loss.detach())

    def state_bytes(self):
        n = sum(p.numel() for p in self.masters)
        return n * 4 * 3  # fp32 masters + Adam exp_avg + exp_avg_sq
