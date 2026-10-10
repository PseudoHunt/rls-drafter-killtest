"""Extra target-side cost of the selection signal per round (T0b): one attention row at the last committed position,
summed over heads and reduced to 32-token chunk scores, for 1 layer (S1) or 3 layers (S2), given the target's KV
cache (already resident during verification). Qwen3-4B shapes: 32 query heads, 8 KV heads, head_dim 128, bf16 cache.
Prints ms per round on the current GPU (median of 200 after warm-up). The offline replay precomputes all rows in the
teacher-forced pass instead (not a deployment cost)."""
import statistics
import time

import torch


def row_scores(q, K, chunk=32):
    Hq, Hkv, d = q.shape[0], K.shape[0], q.shape[-1]
    s = (q.float().reshape(Hkv, Hq // Hkv, d) @ K.float().transpose(-1, -2)).reshape(Hq, -1) * d**-0.5
    p = torch.softmax(s, -1).sum(0)
    T = p.shape[0]
    p = torch.nn.functional.pad(p, (0, -T % chunk))
    return p.view(-1, chunk).sum(-1)


def bench(T, layers, dev):
    qs = [torch.randn(32, 128, dtype=torch.bfloat16, device=dev) for _ in range(layers)]
    Ks = [torch.randn(8, T, 128, dtype=torch.bfloat16, device=dev) for _ in range(layers)]
    ts = []
    for i in range(220):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sum(row_scores(q, K) for q, K in zip(qs, Ks))
        top = out.topk(16).indices
        torch.cuda.synchronize()
        if i >= 20:
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


if __name__ == "__main__":
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    print("| context T | 1 layer (S1/S3) ms | 3 layers (S2) ms |")
    print("|---|---|---|")
    for T in (2048, 8192, 16384):
        print(f"| {T} | {bench(T, 1, dev):.3f} | {bench(T, 3, dev):.3f} |")
