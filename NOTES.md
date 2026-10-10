# NOTES — DFlash inference code read-through (Stage 0)

Reference code: `dflash==0.1.0` from PyPI. Its `dflash/model.py` is byte-identical to z-lab/dflash HEAD
`07ebd93` (2026-08-18). Our harness imports it unmodified: `dflash_generate`, `DFlashDraftModel.forward`,
`compute_logits`, `_make_cache`, `_crop_to`, `_raw_input_embeddings` and `_output_head`.
Line numbers below refer to `dflash/model.py`.

Checkpoint `z-lab/Qwen3-4B-DFlash-b16` (sha `b74e3a3`) has:
- 5 drafter layers, hidden size d = 2560, block size 16;
- `target_layer_ids = [1, 9, 17, 25, 33]` from the 36 target layers;
- `mask_token_id = 151669`.

## 1. Round structure and which slot predicts which token (`dflash_generate`, L180–324)

- **Prefill** (L205–220). The target runs over the prompt (positions 0..P−1). Its argmax at P−1 gives
  `s_P`, the first output token. `target_hidden` holds the context features for every prompt position.
  Start at `start = P`. Positions `< start` have target states; `s_start` is committed but its target
  state is not yet computed.
- **Round with frontier t = start** (L233–308):
  - The block is `[s_t, MASK × 15]` at positions `t..t+15`. Positions after `start` are always still
    MASK, because `output_ids` starts all-MASK and only committed positions are ever written.
  - The drafter input is
    `noise_embedding = target.embed_tokens(block) * input_embedding_scale` (scale = 1 for this checkpoint),
    plus `target_hidden` (context features for positions `c0..t−1`, i.e. only the ones not yet fed),
    with `position_ids = arange(c0, t+16)`.
  - `draft_hidden = model(...)[:, 1-16:, :]`, so **slots i = 1..15 sit at positions t+i**.
  - `d_i = argmax lm_head(z_i)`, and **slot i predicts `s_{t+i}`**.
- **Verify** (L269–292). The target runs on the block (anchor plus 15 drafts) at positions t..t+15.
  - `posterior[j] = argmax` at position t+j, which predicts `s_{t+j+1}`.
  - `acceptance_length = cumprod(block[1:] == posterior[:-1]).sum()`, so **d_i is checked against the
    target argmax at position t+i−1**.
  - The bonus is `posterior[a]`, i.e. the target's prediction at t+a.
  - `produced = a+1`. It is clipped by max_length and by stop tokens (L295–302).
  - `start += produced`.
- **Supervision alignment.** The pair is `(z_i, h*_{t+i−1})`, where `h*_{t+i−1}` is the target's final-norm
  state at position t+i−1. That state's `lm_head` produces the distribution over `s_{t+i}`. Verification
  computes target states at t..t+15. Only t..t+a lie on the true prefix; t+a+1.. condition on rejected
  drafts. So valid pairs are **i = 1..a+1**, which matches brief §4.4.
  - One edge case: when a = 15 (all accepted), i = 16 has no slot. We use `m = min(a+1, 15)`.
- **Decision:** τ_round is DFlash's `produced`. This equals a+1 except in the final round of a trace,
  where it is clipped at EOS or max_length.

## 2. How context features are built and appended

- **Feature layout.** `extract_context_feature(hidden_states, layer_ids)` takes
  `hidden_states[layer_id + 1]` (index 0 is the embeddings, so this is the output of decoder layer
  `layer_id`). It concatenates them, giving `[T, 5 × 2560]`.
  - The replay captures exactly these tensors with forward hooks on `target.model.layers[l]`.
  - `test_hook_features_match_output_hidden_states` checks the hooks against `output_hidden_states`.
- **Inside the drafter** (`DFlashDraftModel.forward`, L583–597):
  - `target_hidden = hidden_norm(fc(features))`.
  - In every layer, the context K/V come from `k_proj` / `v_proj` (plus `k_norm` and RoPE) applied to
    that same projected context, **not** from the previous drafter layer.
  - Queries come only from the 16 noise tokens. They attend, non-causally, to context plus block. There
    is no mask, because `is_causal` is False for full-attention layers.
- **Incremental cache.** The drafter `DynamicCache` receives
  `[ctx K/V of the newly committed positions, block K/V]` each round. It is then cropped to `start`
  (L250), so it holds context K/V for all positions `< t`.
  - After verification, `target_hidden = features(verify output)[:produced]`, which covers positions
    t..t+produced−1. Those are fed next round.
  - The replay does the same, using the teacher-forced features `ctx[c0:t]`.
- **Staleness (TTS).** The cached context K/V were computed with the drafter parameters at the time each
  position was appended. The "refresh" option recomputes all of them with the current parameters
  (`rebuild_context_cache`, which uses the same math as L384–393). A test checks it equals the
  incremental cache when parameters are unchanged.

## 3. Where lm_head is applied

- `compute_logits(draft_hidden, _output_head(target))` (L599–605). The drafter has **no head of its
  own**: it uses the target's `lm_head`, which Qwen3-4B ties to `embed_tokens`.
- For this checkpoint, `output_multiplier = 1` and there is no softcap, so draft logits are
  `lm_head(z)`.
  - **z = drafter output after its final `self.norm`** (L597). It has d = 2560 and lives in the same
    space as the target's `h*` (the target's final-norm output). So `h* − z` and `M z̃ → h*` are
    well-posed.
- Target logits for TTS are `lm_head(h*)`, recomputed from `h*` for the m valid positions only.

## 4. Training position weights

- The DFlash repo has no training code. The DFlash paper (arXiv 2602.06036, §4.2 eq. 4 and §5) gives
  `w_k = exp(−(k−1)/γ)` with **γ = 7 for block 16**.
- TTS re-uses "the weights already incorporated in the training codebase". We take k = slot index
  (k = 1 is the first predicted token).
  - If k instead counted the anchor, every weight would change by the same constant factor e^{−1/7}.
  - Adam is invariant to the loss scale (up to ε), so this choice does not matter for §5.4.

## 5. Implementation decisions (please confirm at Gate A)

1. **Thinking mode vs checkpoint.** The HF card says this drafter "is used for thinking mode disabled"
   (README: "Qwen3 (4B/8B non-thinking)"). No separate thinking-mode Qwen3-4B DFlash checkpoint exists,
   so per the brief I keep thinking ON with `z-lab/Qwen3-4B-DFlash-b16`, which is what TTS used. Expect
   the baseline τ to sit near TTS's ~3.3.
2. **Prompt templates.** DFlash HEAD no longer ships AIME or LCB loaders. I copied them verbatim from
   DFlash's first commit `bb48ef9`, `model/utils.py`:
   - math: `"{problem}\nPlease reason step by step, and put your final answer within \\boxed{}."`;
   - LCB: the `format_lcb` system-prompt, question, format and answer-footer template.
   - Each prompt is a single user turn, put through the Qwen3 chat template with
     `enable_thinking=True`, then `encode(add_special_tokens=False)`, exactly like DFlash's benchmark.
3. **LiveCodeBench v6** = `test6.jsonl` from `livecodebench/code_generation_lite`, i.e. the problems
   added in release_v6 (contests 2025-01-04 to 2025-04-06).
   - It has 80 hard problems. I sort them by `question_id` and take `random.Random(0).sample(hard, 32)`.
   - Dev items are `random.Random(0).sample(range(n), 6)` on each dataset's trace_id-sorted list.
   - AIME dev: II_02, II_09, II_13, II_14, I_10, I_13.
   - LCB dev: 3701, abc390_e, abc393_d, arc191_d, arc193_d, arc195_c.
4. **RLS prior: the brief's objective and recursion disagree when λ < 1.** This needs your decision; see
   §6.
5. **Bias-only.** `b = Σ ρ^age r / Σ ρ^age` is the normalised exponentially weighted mean of the
   residuals `h* − z`.
   - Age is counted in tokens, and tokens within a round are ordered by slot.
   - b is applied to all 15 slots.
6. **RLS / bias input.** The update uses the raw drafter z (not z′) as the regressor. The adapted z′ is
   computed in fp32 and cast back to bf16 before `lm_head`, which is the same dtype path as the
   baseline.
   - I parameterise `M = M0 + ΔM`. This is algebraically the brief's update, and it makes the round-1
     drafts at M0 bit-exact.
   - σ² is the mean of z² over all 15 slots × all rounds of the **dev baseline** runs. The baseline runs
     first; the RLS grid then uses `δ = n0 · σ²`.
   - Forgetting: batched update with `λ_r = λ^m`. Every token in a round shares the age of the round's
     newest token. The unit test checks this against the matching closed-form weighted ridge.
7. **TTS.**
   - Optimiser: Adam with betas (0.9, 0.999) and ε = 1e-8, on **fp32 master copies** of all drafter
     parameters (537M). After each step the masters are copied back into the bf16 drafter used for
     drafting, which keeps the dtype path. Without masters, a single 1e-5 step is below bf16
     resolution for most weights.
   - The gradient comes from re-running this round's drafter forward with grad. Inputs are the same
     context chunk and block; the cached context K/V for earlier positions are constants (stale), as in
     the online system.
   - Loss: `Σ_{i≤m} w_i KL(softmax(lm_head h*_{t+i−1}) ‖ softmax(lm_head z_i))`, summed, in fp32.
   - All drafter state is restored from a pristine copy after each trace.
   - Stale vs refresh: on dev, pick the best stale lr first. Then run refresh at that lr only (one extra
     config). Use refresh on test only if it beats stale by more than 5% τ.
8. **Leakage guard.**
   - `ctx` and `h*` are wrapped (`rlsdk/guard.py`); any read at index ≥ frontier raises.
   - The frontier starts at P and moves to `t + produced` only after verification.
   - Verification itself compares drafts to the committed trace tokens, i.e. the target's greedy
     choices.
9. **Loop flag.** All 32-token windows (stride 1) of the continuation are considered. A trace is flagged
   if more than 50% of windows *starting* in the second half have an identical window starting at an
   earlier position.
10. **Numerics.**
    - TF32 is disabled for all fp32 math.
    - The teacher-forced target pass is a single bf16 SDPA forward over the whole sequence.
    - The real-vs-replay check (Stage 2) measures whether this differs from DFlash's incremental
      verification. The near-tie rate is logged per trace, and the runner warns if it exceeds 1%.

## 6. RLS prior: decision needed

The brief's objective is

    min_M Σ λ^age ‖h* − M z̃‖² + δ‖M − M0‖²_F      (prior does NOT decay)

The brief's recursion, starting from `P0 = I/δ` with `P ← (P − K Zᵀ P)/λ_r`, gives
`P_N⁻¹ = λ^N δ I + Σ λ^age z̃ z̃ᵀ`. **Here the prior decays as λ^N.**

This matters at the brief's λ values. With λ = 0.995 the data window is about 200 tokens, while
d + 1 = 2561. In directions the recent z's don't cover, `P` grows like `λ^{−N}/δ` (classic RLS
covariance windup). After 16K tokens that is e^{82} — fp32 overflow territory.
- λ = 0.999: e^{16}.
- λ = 0.9999: e^{1.6}, which is harmless.

I implemented both versions, and both pass the closed-form test against their own objective:

- `prior: decay`: the literal recursion. Cost is O(d² m) per round: 26 MB for ΔM plus 26 MB for P.
- `prior: fixed`: the exact objective as written. It keeps `G = Σ λ^age z̃z̃ᵀ` and `B`, plus a Cholesky
  factor of `G + δI` per round. Cost is O(d³/3) per round, a few ms on an A100 for d = 2561, using
  78 MB. Still closed-form and backprop-free.

The windup demo results are below. My recommendation: **use `fixed`**. It is the objective the brief
states, it is numerically safe across the whole λ grid, and its cost stays far below a TTS backward
plus Adam step. Alternatively, keep `decay` as the primary method and accept the windup risk at
λ = 0.995. Either way, I'll freeze the choice before any GPU results.

**Windup demo** (CPU, fp32, 16K tokens in rounds of 4, `δ = 16σ²`, synthetic `h = (I + A) z + noise`):

| z distribution | λ | `decay` (literal recursion) | `fixed` (objective as written) |
|---|---|---|---|
| full-rank, power-law spectrum, d = 512 | 0.995 / 0.999 / 0.9999 | residual MSE 0.0070 / 0.0032 / 0.0026 | 0.0059 / 0.0031 / 0.0026 |
| **rank-96 subspace in d = 256**, then probed with a small off-subspace shift | 0.995 | **NaN** (P overflowed) | 0.0055 (no adaptation: 0.144) |
| same | 0.999 | **NaN** | 0.0049 |

Real drafter outputs are strongly anisotropic (RMSNorm outputs of an LLM), so the rank-deficient case is
the relevant one. With `decay`, the λ = 0.995 and 0.999 grid cells could produce NaN or garbage drafts;
that would be a numerical artefact, not a property of the method.

## 7. GPU-time plan (refined after Stage 2 timing)

| Stage | Work | Estimate |
|---|---|---|
| 1 | vLLM, 62 traces × ≤16K | ~0.3–0.6 h |
| 2 | 3 × 4K real DFlash generate + replay | ~0.2 h |
| 3 | 12 dev traces × (1 base + 3 bias + 9 RLS + 3 TTS + 1 refresh) | ~1.5–2.5 h, mostly TTS |
| 4 | 50 test traces × 4 frozen configs | ~1.5–2 h |

The teacher-forced pass runs once per trace and is shared across all configs.

## 8. Gate A resolutions (see KILL_TEST_BRIEF.md §11)

- RLS arm = `prior: fixed` (exact solver) for λ < 1, and the recursion (`prior: decay`) at λ = 1 (the default in
  `rlsdk/adapters.py::RLS`). Grid: n0 ∈ {16, 128, 1024} × λ ∈ {0.995, 0.999, 0.9999, 1.0}.
- fp32 solve, with an automatic fp64 fallback on Cholesky failure (logged in the header as
  `diagnostics.fp64_fallback_at`). `scripts/rls_checks.py` runs on one dev trace: λ = 1 recursion vs fixed
  (rel. err. of M < 1e-4), and fp32 vs fp64 corrected logits (> 1e-3 → fp64 for all RLS runs). The fp64 check uses
  the worst-conditioned cell (n0 = 16, λ = 0.995).
- Per-round `solve_ms` = time in the Cholesky factorisation and solves (fixed) or the recursion's update algebra
  (decay). It is contained in `draft_ms + update_ms`.
- TTS refresh: at the best stale lr; if it wins by > 5% there, also run it at the other two lrs.
- Thinking-OFF control: `traces/traces_nothink.jsonl` (6 dev AIME prompts), baseline replay only.

## 9. How DFlash (dflash 0.1.0) applies RoPE to context keys — for the windowed-context test

- Each drafter layer (`Qwen3DFlashAttention.forward`, 5 layers, 8 KV heads × 128) builds keys from **two sources**:
  `k_ctx = k_proj(hidden_norm(fc(target_hidden)))` for the newly committed context rows and `k_noise = k_proj(block)`
  for the 16 block slots. Both go through `k_norm`, are concatenated, and are rotated **together** by
  `apply_rotary_pos_emb(q, k, cos, sin)` with `cos/sin = rotary_emb(position_ids)` (θ = 1e6, no scaling). Queries
  use only the last `q_len` rows of cos/sin.
- Replay/generate pass `position_ids = arange(c0, start + 16)`: **absolute positions**. New context rows get their
  true positions; block slots get `start .. start + 15`.
- The keys are rotated **before** `past_key_values.update(k, v)`, so the `DynamicCache` stores *post-RoPE* keys at
  absolute positions. After each forward the cache is cropped to `start` (drops the 16 block keys), so the cache
  holds every context position `0 .. start − 1`, each rotated once with its own absolute position.
- Attention is **full / non-causal** (`is_causal: null`, all layers `full_attention`, no sliding window, no mask):
  every block query sees every cached context key and every block key. The largest query–key distance is therefore
  `start + 15`, i.e. it grows with output position (up to ≈ 17K on our traces; `max_position_embeddings` 40960).
- Consequence for a StreamingLLM-style window: the cached keys cannot simply be sliced, because their rotation encodes
  absolute positions. `rlsdk/window.py` stores context keys **pre-RoPE** (post-`k_norm`), keeps
  {kept set} ∪ {last W context positions}, and every round rotates them at **contiguous** positions
  `0 .. L − 1` (L = |kept| + |window|), with the block at `L .. L + 15`. Max query–key distance = L + 15 ≤ |kept| + W + 15.
  When the whole context fits (start ≤ |kept| + W), positions equal the absolute ones and the computation is the
  baseline's, element for element (bit-exact; unit-tested). Target pass and verification are unchanged.

## 10. Mechanism + selection screen (context policies B, E, C, C′, D)

All implemented in `rlsdk/window.py` (one cache: pre-RoPE context keys, per-round visible set + key positions, and an
optional top-k mask applied inside the attention call); target, verification and eval unchanged. Key budget 1024
wherever keys are selected.

- **B** `winorig` (W = 1024, keep prompt): the best window's visible set, but every key and query at its original
  absolute position (no renumbering). The kept prompt already sits at 0 .. P−1 in the contiguous scheme, and the
  window is adjacent to the block in both schemes, so the *only* difference from the window is the query↔prompt
  distance: ≈ start (original) vs ≈ P + W (contiguous).
- **E** `capctx` (cap = 4096): all context keys. Keys at positions ≥ start − 4096 keep their positions; every older
  key is rotated at start − 4096 (anchor = start, the block's first position), so query i sees all far keys at
  distance 4096 + i (ReRoPE/Self-Extend-style clipping). All keys are kept (no key budget).
- **C** `topk` (k = 1024): all context keys at original positions; per layer, per query head, per query, keep the
  1024 context keys with the largest logit q·k/√d (fp32), block keys always kept, softmax over the kept set.
- **C′** `topkcap` (k = 1024, cap = 4096): as C, but logits and attention under E's capped rotation.
- **D** `topkshared` (k = 1024, cap = 4096): as C′, but per query head the logits are **summed over the 16 block
  queries** and one top-1024 set is used for every query of the block. ("Scores" = logits, not probabilities.)
- GQA: 32 query heads share 8 KV heads; selection is per query head (head h uses KV head h // 4, as `repeat_kv`).
- Exactness: with W/cap/k ≥ context length nothing is dropped, moved or masked, and the forward is the baseline's
  element for element (tests/test_ctxpolicy.py; E with cap = 1e6 on lcb_abc393_d: 2,675/2,675 rounds identical on
  the A30). With selection active the masked SDPA (math/efficient kernel) replaces the unmasked kernel.

## 11. T0b: target-guided far-context selection

- Signal: after each verification the next round starts at t' (= `start`). The target's attention row at the last
  committed position t'−1 is summed over its 32 query heads and reduced to 32-token chunks, A[t'−1, c] =
  Σ_heads Σ_{key ∈ [32c, 32c+32)} softmax(q_{t'−1}·k / √d). It depends only on tokens ≤ t'−1. Layers (0-based):
  `last` = 35; `mid3` = mean of 17, 25, 33 (the layers DFlash reads). Refreshed every round.
- Replay: all rows are computed once per trace inside the teacher-forced pass (`rlsdk/target_attn.py`: pre-hook on
  each scoring layer's `self_attn` captures its input, q/k rebuilt as in `Qwen3Attention`, causal softmax in fp32,
  query blocks of 256) and read through the leakage guard (`Guarded`, frontier = t'): only row t'−1 is read
  (tested). Matches eager attention on the tiny model (atol 1e-5).
- Far region = after the kept prompt and older than the window, [P, t' − W); candidates = aligned 32-token chunks
  lying wholly inside it. Keep the top 16 by score (ties by `torch.topk`), in original order. Visible keys = prompt
  + chosen chunks + window, renumbered contiguously (as the window: positions 0 .. L−1, block at L .. L+15).
- Controls: C1 = plain window W 1536 + prompt (same key count as S1: 1024 + 512); C2 = W 1024 + prompt + 16
  random far chunks, re-drawn every round from a CPU generator with seed 0 (deterministic).
- Deployment cost: one extra attention row per round from the target's resident KV cache (microbenchmark in
  `results/attn_row_cost.md`); the replay's full precompute is an offline convenience, not a deployment cost.

## 12. Verifier-Prior Attention (VPA; held-out addendum B)

- Kept context as S2 (prompt + last W0 = 1024 + top M far 32-token chunks by S2's chunk scores, contiguous
  renumbering), plus an additive bias on the drafter's attention logits for the kept CONTEXT keys, in every drafter
  layer and head (block keys 0): b_j = β·log(m_j + 1/N_ctx) − λ·log(1 + max(0, d_j − W0)/W0), distance term on far-chunk
  keys only, centred over the kept context keys (so a constant offset is a no-op and the context/block split is not
  shifted by a uniform term). β = λ = 0 skips the bias entirely (bit-exact S2 / window).
- m_j: the target's attention mass on context position j from the last committed query t'−1: post-RoPE q row and K
  rows of target layers 17/25/33 are stored per position in the teacher-forced pass (`target_attn.qk_store`, read
  through the leakage guard: only q[t'−1] and K[0 .. t'−1]); per round softmax per head (fp32), summed over the 32
  heads, mean over the 3 layers, normalised to sum 1 over 0 .. t'−1 (`target_attn.key_mass`; = eager attention, tested).
- The bias enters SDPA as an additive mask [1, 1, Q, K] (drafter dtype) through the same attention hook as top-k
  (`window.topk_attention`); unmasked SDPA otherwise.
