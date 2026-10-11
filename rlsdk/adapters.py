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


class Window(Baseline):
    """No adaptation; the drafter's context is windowed (rlsdk/window.py). config: W, keep (int or "prompt")."""

    method = "window"

    def __init__(self, config):
        super().__init__(config)
        keep = config["keep"]
        self.ctx_window = {"W": int(config["W"]), "keep": keep if keep == "prompt" else int(keep)}


class FarSel(Baseline):
    """No adaptation; window + kept prompt + n far 32-token chunks re-selected every round (T0b, NOTES.md §11).
    config: W, keep, n, score ("last" | "mid3": target attention row at start-1; "random": fixed seed), seed."""

    method = "farsel"

    def __init__(self, config):
        super().__init__(config)
        keep = config["keep"]
        self.ctx_window = {"W": int(config["W"]), "keep": keep if keep == "prompt" else int(keep),
                           "positions": "contiguous",
                           "far": {"n": int(config["n"]), "score": config["score"], "seed": int(config.get("seed", 0))}}


class VPA(Baseline):
    """Verifier-Prior Attention (NOTES.md §12): S2-style kept context (prompt + last W0 + top M far chunks by the
    target layers 17/25/33 signal) plus an additive bias on the kept context keys' attention logits.
    config: M, beta, lam, W0 (default 1024), keep (default "prompt"), b_offset (tests only)."""

    method = "vpa"

    def __init__(self, config):
        super().__init__(config)
        keep = config.get("keep", "prompt")
        W0 = int(config.get("W0", 1024))
        vpa = {"beta": float(config["beta"]), "lam": float(config["lam"]), "W0": W0}
        if "b_offset" in config:
            vpa["b_offset"] = float(config["b_offset"])
        self.ctx_window = {"W": W0, "keep": keep if keep == "prompt" else int(keep), "positions": "contiguous",
                           "far": {"n": int(config["M"]), "score": "mid3", "seed": 0, "vpa": vpa}}


class CtxPolicy(Baseline):
    """No adaptation; drafter context policy (rlsdk/window.py, NOTES.md §10). The method name sets the mechanism:
      winorig     kept set + last W positions, original position ids        config: W, keep
      capctx      all context keys, positions older than start-cap at start-cap     config: cap
      topk        all keys, original positions, top-k per head per query     config: k
      topkcap     all keys, capped positions, top-k per head per query        config: k, cap
      topkshared  all keys, capped positions, top-k per head shared over the block (summed logits)  config: k, cap
    """

    MODES = {"winorig": "original", "capctx": "capped", "topk": "original", "topkcap": "capped", "topkshared": "capped"}

    def __init__(self, method, config):
        super().__init__(config)
        self.method = method
        keep = config.get("keep")
        self.ctx_window = {"positions": self.MODES[method], "W": config.get("W"),
                           "keep": keep if keep in (None, "prompt") else int(keep), "cap": config.get("cap"),
                           "topk": config.get("k"), "shared": method == "topkshared"}


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


TTS_CTX = {  # drafter context policies for TTS (P3 arms); same dicts as the frozen window / S2 arms
    "window": {"W": 1024, "keep": "prompt", "positions": "contiguous"},
    "s2": {"W": 1024, "keep": "prompt", "positions": "contiguous", "far": {"n": 16, "score": "mid3", "seed": 0}},
}


class TTS:
    """Adam on all drafter parameters (§5.4).  fp32 master weights + Adam state; the bf16 drafter used for drafting
    is refreshed from the masters after every step.  The actual gradient computation needs the drafter graph, so
    `update` is driven by replay.py via `rnd`.

    config: lr; refresh (stock cache only); stride k (default 1): one Adam step every k updates (rounds with valid
    pairs); pool: False -> the step uses only the latest round's loss (no gradient on the other rounds); True -> the
    step uses the summed loss of all k rounds' valid pairs (gradients accumulated in fp32 on the masters, each round's
    term under that round's drafter context); ctx: None (full context) | "window" | "s2" (TTS_CTX): the drafter
    drafts and back-propagates with that context policy only.
    """

    method = "tts"

    def __init__(self, config):
        self.config = config
        self.lr = float(config["lr"])
        self.refresh = bool(config.get("refresh", False))
        self.gamma = float(config.get("gamma", 7.0))
        self.stride = int(config.get("stride", 1))
        self.pool = bool(config.get("pool", False))
        # batched: exact batched pooled update (rlsdk/batched.py): store each round's loss inputs, one batched
        # forward + backward at the step. check (tests): also accumulate the per-round gradient and log the rel. error
        self.batched = bool(config.get("batched", False))
        self.check = bool(config.get("check", False))
        # warm (output tokens): rounds starting before output position `warm` take a per-round Adam step (stride 1);
        # later rounds use the batched pooled schedule (stride counted from the first post-warm round)
        self.warm = None if config.get("warm") is None else int(config["warm"])
        if self.warm is not None and not self.batched:
            raise ValueError("warm is implemented for the batched pooled schedule only")
        ctx = config.get("ctx")
        if self.batched and not (self.pool and ctx is not None):
            raise ValueError("batched needs pool: true and a window / S2 ctx")
        if ctx is not None:
            self.ctx_window = dict(TTS_CTX[ctx])
            if self.refresh:
                raise ValueError("refresh is implemented for the stock cache only")

    def bind(self, drafter, pristine_state):
        self.drafter = drafter
        self.pristine = pristine_state

    def reset(self, d, device):
        self.drafter.load_state_dict(self.pristine)
        self.params = [p for p in self.drafter.parameters()]
        for p in self.params:
            p.requires_grad_(True)
        self.masters = [p.detach().float().clone() for p in self.params]
        # fused Adam on CUDA (Amendment §12.3): same update rule, one kernel
        self.opt = torch.optim.Adam(self.masters, lr=self.lr, betas=(0.9, 0.999),
                                    fused=self.masters[0].device.type == "cuda")
        self.weights = dflash_position_weights(16, self.gamma).to(device)
        self.n_upd = 0
        self.n_steps = 0
        self._log = {}
        self.buf, self.ref = [], None
        self.n_pool = 0
        self.check_err = []

    def pop_round_log(self):
        out, self._log = self._log, {}
        return out

    def diagnostics(self):
        d = {"stride": self.stride, "pool": self.pool, "n_updates": self.n_upd, "n_steps": self.n_steps}
        if self.batched:
            d["batched"] = True
        if self.check:
            d["check_rel_err"] = self.check_err
        return d

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
        self.n_upd += 1
        step_now = self.n_upd % self.stride == 0
        if self.batched:
            if self.warm is not None and rnd.pos < self.warm:
                loss = self._run_det(lambda: self._grad(m, rnd))
                self._step()
                self._log = {"grad": 1, "step": 1}
                return loss
            self.n_pool += 1
            return self._update_batched(m, rnd, self.n_pool % self.stride == 0)
        if not (step_now or self.pool):
            self._log = {"grad": 0, "step": 0}
            return None
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            loss = self._grad(m, rnd)
            if step_now:
                self._step()
        finally:
            torch.use_deterministic_algorithms(prev_det)
        self._log = {"grad": 1, "step": int(step_now)}
        if self.refresh and step_now:
            rnd.refresh_cache()
        return loss

    def _grad(self, m, rnd, into=None):
        """Back-propagate this round's loss; accumulate into the fp32 master grads (or into the list `into`)."""
        with torch.enable_grad():
            zg = rnd.grad_forward()[:m]
            q_logp = torch.log_softmax(rnd.draft_logits(zg).float(), dim=-1)
            p_logp = torch.log_softmax(rnd.target_logits.float(), dim=-1)
            kl = (p_logp.exp() * (p_logp - q_logp)).sum(-1)  # KL(p || q) per slot
            loss = (self.weights[:m] * kl).sum()
            loss.backward()
        self._accumulate(into)
        return float(loss.detach())

    def _accumulate(self, into=None):
        for i, (p, mp) in enumerate(zip(self.params, self.masters)):
            if p.grad is not None:
                g = p.grad.float()
                if into is None:
                    mp.grad = g if mp.grad is None else mp.grad + g
                else:
                    into[i] = g if into[i] is None else into[i] + g
            p.grad = None

    def _update_batched(self, m, rnd, step_now):
        from .batched import batched_loss
        self.buf.append(rnd.capture())
        if self.check:  # reference: the per-round accumulated gradient (same trajectory)
            if self.ref is None:
                self.ref = [None] * len(self.params)
            self._run_det(lambda: self._grad(m, rnd, into=self.ref))
        if not step_now:
            self._log = {"grad": 0, "step": 0}
            return None
        loss = None

        def go():
            nonlocal loss
            loss = 0.0
            for q in sorted({r.q for r in self.buf}):  # one batch per block size (smaller only at max_length)
                recs = [r for r in self.buf if r.q == q]
                with torch.enable_grad():
                    lt = batched_loss(self.drafter, rnd.head, recs, self.weights)
                    lt.backward()
                self._accumulate()
                loss += float(lt.detach())
            if self.check:
                num = sum(float(((mp.grad - g) ** 2).sum()) for mp, g in zip(self.masters, self.ref)
                          if g is not None)
                den = sum(float((g ** 2).sum()) for g in self.ref if g is not None)
                self.check_err.append((num / den) ** 0.5)
                self.ref = None
            self._step()

        self._run_det(go)
        self.buf = []
        self._log = {"grad": 1, "step": 1}
        return loss

    def _run_det(self, fn):
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return fn()
        finally:
            torch.use_deterministic_algorithms(prev_det)

    def _step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.n_steps += 1
        with torch.no_grad():
            for p, mp in zip(self.params, self.masters):
                p.copy_(mp)

    def state_bytes(self):
        n = sum(p.numel() for p in self.masters)
        return n * 4 * 3  # fp32 masters + Adam exp_avg + exp_avg_sq


_W32 = {}  # one fp32 copy of the (frozen) lm_head weight per process, shared by KL adapters


def _w32(head):
    w = head.weight
    key = (w.data_ptr(), w.device, tuple(w.shape))
    if key not in _W32:
        _W32.clear()
        _W32[key] = w.detach().float()
    return _W32[key]


def _rms(x, dim=None):
    return (x * x).mean(dim=dim, keepdim=dim is not None).sqrt()


class _KLBase:
    """Shared plumbing for the KL arms: W (fp32 lm_head weight, tied embeddings) is bound by replay.simulate;
    p_i = softmax(W h*_{t+i-1}), q_i = softmax(W z_i) with z_i the adapted (drafting) z; slot weights
    w_i = exp(-(i-1)/7). The drafter has no output multiplier / softcap, so draft logits = W z."""

    gamma = 7.0

    def bind_head(self, head):
        self.W = _w32(head)

    def reset(self, d, device):
        self.weights = dflash_position_weights(16, self.gamma).to(device)
        self._zp = None
        self._log = {}

    def _p_minus_q(self, zp, h, slots):
        """Rows (w_i (p_i - q_i)) for the valid slots, [m, V] fp32."""
        p = torch.softmax(h.to(self.W.dtype) @ self.W.T, dim=-1)
        q = torch.softmax(zp @ self.W.T, dim=-1)
        return self.weights[slots - 1][:, None] * (p - q)

    def pop_round_log(self):
        out, self._log = self._log, {}
        return out


class KLBias(_KLBase):
    """Reflex-form KL bias, no gate.  g = sum_{i<=a+1} w_i W^T (p_i - q_i);  m <- rho m + (1-rho) g/RMS(g);
    z' = z + alpha RMS(z) m/RMS(m) (RMS(z) per slot row)."""

    method = "klbias"

    def __init__(self, config):
        self.config = config
        self.rho = float(config.get("rho", 0.9))
        self.alpha = float(config["alpha"])

    def reset(self, d, device):
        super().reset(d, device)
        self.mom = torch.zeros(d, dtype=torch.float32, device=device)
        self.n_updates = 0

    def apply(self, z, slots):
        zf = z.float()
        if self.n_updates == 0 or self.alpha == 0.0:
            self._zp = zf
            return z
        rm = _rms(self.mom)
        zp = zf if rm == 0 else zf + self.alpha * _rms(zf, -1) * (self.mom / rm)
        self._zp = zp
        return zp.to(z.dtype)

    def update(self, z, h, slots, rnd):
        m = z.shape[0]
        g = (self._p_minus_q(self._zp[:m], h, slots) @ self.W).sum(0)  # [d]
        rg = _rms(g)
        if rg > 0:
            self.mom = self.rho * self.mom + (1 - self.rho) * g / rg
        self.n_updates += 1

    def state_bytes(self):
        return self.mom.numel() * 4


class KLLinear(_KLBase):
    """z' = z + A z~,  A in R^{d x (d+1)} from 0.  Loss sum_{i<=a+1} w_i KL(p_i || softmax(W z'_i));
    closed-form gradient grad_A = sum_i w_i W^T (q'_i - p_i) z~_i^T (no autograd through the drafter).
    Adam (betas 0.9/0.999, eps 1e-8), A and Adam state fp32.  Logs rel = max_i ||A z~_i|| / ||z_i|| per round."""

    method = "kllinear"
    REL_FLAG = 0.2

    def __init__(self, config):
        self.config = config
        self.lr = float(config["lr"])

    def reset(self, d, device):
        super().reset(d, device)
        self.A = torch.zeros(d, d + 1, dtype=torch.float32, device=device)
        self.opt = torch.optim.Adam([self.A], lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.n_flagged = 0
        self.max_rel = 0.0

    @staticmethod
    def _tilde(zf):
        return torch.cat([zf, torch.ones(zf.shape[0], 1, dtype=zf.dtype, device=zf.device)], dim=1)

    def apply(self, z, slots):
        zf = z.float()
        corr = self._tilde(zf) @ self.A.T
        rel = float((corr.norm(dim=-1) / zf.norm(dim=-1).clamp_min(1e-30)).max())
        self._log = {"rel": round(rel, 6)}
        if rel > self.REL_FLAG:
            self.n_flagged += 1
        self.max_rel = max(self.max_rel, rel)
        self._zp = zf + corr
        return self._zp.to(z.dtype)

    def grad(self, zf, h, slots, zp):
        """Closed-form dLoss/dA for rows zf (un-adapted z), zp = adapted z."""
        G = -self._p_minus_q(zp, h, slots) @ self.W  # rows w_i W^T (q'_i - p_i)  [m, d]
        return G.T @ self._tilde(zf)

    def update(self, z, h, slots, rnd):
        m = z.shape[0]
        self.A.grad = self.grad(z.float(), h, slots, self._zp[:m])
        self.opt.step()
        self.A.grad = None

    def diagnostics(self):
        return {"rel_flag": self.REL_FLAG, "n_rounds_rel_gt_flag": self.n_flagged, "max_rel": round(self.max_rel, 6)}

    def state_bytes(self):
        return self.A.numel() * 4 * 3  # A + Adam exp_avg + exp_avg_sq (fp32); the fp32 W copy is not counted
