"""Target attention summaries for target-guided context selection (T0b, NOTES.md §11).

For selected target layers, A[q, c] = sum over the 32 query heads of the target's causal attention probabilities
from query position q to the keys in chunk c = [32c, 32c + 32) (keys > q get 0). Computed once per trace from the
teacher-forced pass (layer inputs captured by a pre-hook, q/k rebuilt exactly as Qwen3Attention does, softmax in
fp32, query blocks of 256). In replay only row t'-1 is read, through the leakage guard.
"""
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

CHUNK = 32


class AttnCapture:
    """Pre-hooks on target.model.layers[l].self_attn capturing (hidden_states, position_embeddings)."""

    def __init__(self, target, layers):
        self.target, self.layers, self.inputs, self.hooks = target, list(layers), {}, []

    def __enter__(self):
        for l in self.layers:
            attn = self.target.model.layers[l].self_attn

            def pre(mod, args, kwargs, l=l):
                self.inputs[l] = (kwargs["hidden_states"], kwargs["position_embeddings"])

            self.hooks.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def qk_of(attn, hidden, pos_emb):
    """Post-RoPE q [1, Hq, T, d] and k [1, Hkv, T, d], exactly as Qwen3Attention builds them."""
    T = hidden.shape[1]
    shape = (1, T, -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
    cos, sin = pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)


@torch.no_grad()
def chunk_scores(attn, hidden, pos_emb, chunk=CHUNK, qblock=256):
    """[T, ceil(T/chunk)] fp32 head-summed causal attention mass per key chunk, for one Qwen3Attention layer."""
    T = hidden.shape[1]
    q, k = qk_of(attn, hidden, pos_emb)
    Hq, Hkv, d = q.shape[1], k.shape[1], q.shape[-1]
    g = Hq // Hkv
    nC = (T + chunk - 1) // chunk
    out = torch.zeros(T, nC, dtype=torch.float32, device=hidden.device)
    kf = k[0].float()  # [Hkv, T, d]
    for a in range(0, T, qblock):
        b = min(a + qblock, T)
        qb = q[0, :, a:b].float().reshape(Hkv, g * (b - a), d)  # head h -> kv head h // g (repeat_kv layout)
        s = (qb @ kf[:, :b].transpose(-1, -2)).reshape(Hq, b - a, b) * attn.scaling
        causal = torch.arange(b, device=s.device)[None, :] > torch.arange(a, b, device=s.device)[:, None]
        p = torch.softmax(s.masked_fill(causal, float("-inf")), dim=-1).sum(0)  # [b-a, b]
        nb = (b + chunk - 1) // chunk
        p = torch.nn.functional.pad(p, (0, nb * chunk - b))
        out[a:b, :nb] = p.view(b - a, nb, chunk).sum(-1)
    return out


def default_spec(target):
    """Scoring signals -> target layers (0-based). Qwen3-4B (36 layers): last = 35, mid3 = 17, 25, 33 (the target
    layers DFlash also reads). Other depths (tiny tests): last layer and three spread layers."""
    n = len(target.model.layers)
    if n == 36:
        return {"last": [35], "mid3": [17, 25, 33]}
    return {"last": [n - 1], "mid3": [n // 4, n // 2, n - 2]}


def summaries(target, captured, spec):
    """{name: mean over spec[name] of A_l} for chunk-score signals; spec["qk"] (if present) is handled by qk_store."""
    sig = {n: ls for n, ls in spec.items() if n != "qk"}
    need = sorted({l for ls in sig.values() for l in ls})
    A = {l: chunk_scores(target.model.layers[l].self_attn, *captured[l]) for l in need}
    return {name: sum(A[l] for l in ls) / len(ls) for name, ls in sig.items()}


@torch.no_grad()
def qk_store(target, captured, layers):
    """Per-position post-RoPE q and k of `layers`, position-major so the leakage guard can index them:
    q [T, L, Hq, d], k [T, L, Hkv, d] (model dtype)."""
    qs, ks = zip(*(qk_of(target.model.layers[l].self_attn, *captured[l]) for l in layers))
    return {"q": torch.stack([q[0].transpose(0, 1) for q in qs], dim=1),
            "k": torch.stack([k[0].transpose(0, 1) for k in ks], dim=1), "layers": list(layers)}


def key_mass(q_row, K, scaling):
    """Target attention mass on each context position from one query row: q_row [L, Hq, d], K [N, L, Hkv, d] ->
    [N] fp32, softmax per head, summed over heads, mean over layers, normalised to sum 1."""
    L, Hq, d = q_row.shape
    Hkv = K.shape[2]
    out = 0
    for l in range(L):
        s = (q_row[l].float().reshape(Hkv, Hq // Hkv, d) @ K[:, l].float().permute(1, 2, 0)) * scaling  # [Hkv, g, N]
        out = out + torch.softmax(s, dim=-1).sum((0, 1))
    m = out / (L * Hq)
    return m / m.sum()
