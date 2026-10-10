# Kill test: closed-form (RLS) test-time drafter adaptation vs TTS

## 0. How to work on this

- You are running on a GPU instance. Read this whole brief before doing anything.
- Work in stages (§7). At each **GATE**, stop, report, and wait for my go-ahead.
- **GPU budget: 6 GPU-hours total.** If your projection goes above 8, stop and tell me.
- The instance can be reclaimed without warning:
  - commit and push code after every stage;
  - write all outputs under `$PERSIST` (ask me for the path if it is unset);
  - make every step resumable.

**Background.** Test-Time Speculation (TTS, arXiv:2605.09329) shows DFlash and EAGLE-3 acceptance length decays over long outputs. It fixes this with one online gradient step on the drafter per round, using the target's distributions from verification. We test whether a closed-form, backprop-free update gets most of that gain. The update is recursive least squares (RLS) on the drafter's output hidden state.

## 1. Question and decision rule (fixed before any results)

**Primary metric.** τ = mean committed tokens per verification round (accepted draft tokens + 1 bonus).

- Compute it on the **test split**, pooled over all rounds of all non-looping traces in each dataset.
- Then average the two datasets.

**Gain recovered.** R = (τ_RLS − τ_base) / (τ_TTS − τ_base)

| Outcome | Rule |
|---|---|
| **GO** | R ≥ 0.50 |
| **GREY** | 0.25 ≤ R < 0.50: run exactly one follow-up (§5.5), then apply the rule again |
| **STOP** | R < 0.25 |
| **INVALID** | τ_TTS < 1.10 × τ_base on test: the TTS baseline is too weak to judge against. Report and stop. |

Do not change thresholds, metrics, splits or grids after seeing any results.

## 2. Setup

- **Target:** `Qwen/Qwen3-4B`, bf16.
- **Drafter:** `z-lab/Qwen3-4B-DFlash-b16`. Block 16 means 1 anchor + 15 draft slots. TTS used this checkpoint.
- **Reference code:** https://github.com/z-lab/dflash (`pip install dflash`).
  - Use it for the drafter forward, context-feature extraction, the drafter KV cache and the acceptance logic.
  - Do not edit it in place. Wrap it or add hooks.
- **Environments:** if vLLM and dflash need conflicting versions, use two venvs (uv): `env-vllm` for trace generation, `env-dflash` for everything else.
- **Hardware:** one GPU with ≥ 40 GB.
- **Decoding:**
  - greedy (temperature 0) everywhere;
  - thinking mode ON (Qwen3 default chat template), because we need long outputs;
  - `max_new_tokens = 16384`.
- If the DFlash repo documents a separate thinking-mode checkpoint, ask me before switching.

## 3. Data

**Datasets**

- **AIME 2024:** `MathArena/aime_2024_I` + `MathArena/aime_2024_II` (30 problems).
- **LiveCodeBench v6, hard:** `livecodebench/code_generation_lite`, difficulty = hard. Sample 32 problems with seed 0.
- **Prompt templates:** copy exactly from the DFlash repo's eval code (the math template and the LCB template).

**Split (fixed, seed 0)**

- **dev** = 6 AIME + 6 LCB.
- **test** = the remaining 24 AIME + 26 LCB.
- Tune only on dev.

**Loop filter**

- Flag a trace as looping if more than 50% of the 32-token windows in its second half already appeared earlier in the trace.
- Exclude flagged traces from the decision.
- Report how many were flagged, and the results with them included.

## 4. Replay design

Under greedy decoding, speculative decoding commits exactly the target's greedy trace. So we can replay rounds offline from the trace plus one teacher-forced target pass. This gives the same acceptance lengths real speculative decoding would see.

Notation: `s = [prompt; continuation]`, absolute positions. Each round has an anchor `s_t` and frontier `t`. Context features exist for positions `< t`.

**1. Teacher-forced target pass.** Run the target over `s` (in `env-dflash`, HF). Capture with hooks, for every position j:
- (a) the target layer hidden states DFlash uses, in the exact form the drafter consumes;
- (b) `h*_j` = the target's final hidden state after the final norm (the lm_head input);
- (c) `argmax_j` of the target logits.

Keep these for the current trace only. Do not write features to disk.

**2. Near-tie rate.** Log the fraction of positions where `argmax_j ≠ s_{j+1}`. If it is above 1%, stop and report.

**3. Simulate rounds** exactly as DFlash's inference loop does. Read the code for indexing, anchor and bonus handling, and how context features are appended. For each round:
- The drafter forward gives a per-slot hidden state `z_i` (the lm_head input) and a draft token `d_i = argmax lm_head(z_i)`, for i = 1..15. Slot i predicts `s_{t+i}`.
- Accepted count: `a` = the length of the longest prefix with `d_i == s_{t+i}`.
- `τ_round = a + 1`, and the next frontier is `t' = t + a + 1`.
- Append context features for the newly committed positions using DFlash's incremental cache. Never recompute the full context each round.

**4. Valid supervision after a round.** Use only pairs `(z_i, h*_{t+i−1})`, and target logits at `t+i−1`, for **i = 1..a+1**. Real verification only reveals target states on the true prefix up to the first rejection. Confirm the slot ↔ position alignment against the DFlash code.

**5. Leakage guard.**
- Wrap target-side arrays so any read at index ≥ the current frontier raises.
- The frontier moves to `t' = t + a + 1` only after the round's verification.
- Unit-test this guard.

**6. Per-round log fields.** `trace_id, method, config, round, t, a, tau, draft_ms, update_ms`. Bin rounds by output position into 2K-token bins.

## 5. Methods

These rules apply to every method:
- All adaptation state resets at the start of each trace.
- Updates run after verification and before the next round's drafting.
- Adaptation math runs in fp32. Draft logits use the same dtype path for all methods.

### 5.1 Baseline
The unmodified drafter.

### 5.2 Bias-only (Reflex-like)
- `z' = z + b`, where `b` is an exponentially weighted mean of the residuals `(h* − z)` with decay ρ.
- Grid: ρ ∈ {0.9, 0.99, 0.999}.

### 5.3 RLS (the method)

**Form.** `z' = M z̃`, where `z̃ = [z; 1]` and `M ∈ R^{d×(d+1)}`.
- Initialize `M_0 = [I | 0]`, so the first round reproduces the baseline exactly.
- d = the drafter's lm_head input size. Read it from the config.

**Objective.**
```
min_M  Σ_tokens λ^age ‖h* − M z̃‖²  +  δ ‖M − M_0‖²_F
```

**Batched update.** Each round has m = a+1 valid pairs, with `Z ∈ R^{(d+1)×m}` and `H ∈ R^{d×m}`:
```
K = P Z (λ_r I_m + Zᵀ P Z)⁻¹
M ← M + (H − M Z) Kᵀ
P ← (P − K Zᵀ P) / λ_r
```
- Start with `P_0 = I/δ`.
- Use per-token forgetting: either λ_r = λ^m, or apply the update sequentially. Unit-test that your choice matches the closed-form weighted-ridge solution.
- Symmetrize P every 64 rounds.

**Hyperparameters.**
- `δ = n0 · σ²`, where σ² is the mean per-dimension second moment of z on dev traces.
- Grid: n0 ∈ {16, 128, 1024} × λ ∈ {0.995, 0.999, 0.9999}.

### 5.4 TTS-style gradient baseline

**Update.** After each round, take one Adam step on all drafter parameters with loss:
```
Σ_{i ≤ a+1} w_i · KL( p_target(· | t+i−1) ‖ q_draft(· | slot i) )
```
- `w_i` are DFlash's training position weights. Read them from the repo.
- Stride 1, which gives TTS's best acceptance.
- Grid: lr ∈ {1e-5, 3e-5, 1e-4}, Adam betas (0.9, 0.999).
- Omit TTS's second KL term (to the pre-update drafter). It has zero gradient for a single step.

**Context-cache staleness.** Updates change the projections that built the cached context K/V.
- Default: keep the stale cache.
- On dev, also run with a full context-cache refresh after each update.
- If refresh beats stale by more than 5% τ, use refresh on test. Report which one you used.

### 5.5 Per-slot RLS (only if §1 returns GREY)
- Same as 5.3, but with separate (M, P) for slot groups {1}, {2–4} and {5–15}.
- Reuse the frozen λ and n0 from 5.3. Do no new tuning.

## 6. Engineering rules

**CPU first.** Build and test the replay on a tiny model through the same code paths:
- a random-weight tiny Qwen3 config;
- a randomly initialized drafter of the DFlash class with tiny dims.

No GPU use before Gate A.

**Unit tests (pytest), all passing before Gate A:**
1. The RLS batched update matches the closed-form weighted-ridge solution on random data (relative error < 1e-4).
2. With RLS at `M_0`, the first round reproduces baseline drafts bit-exactly.
3. The leakage guard raises on a planted out-of-frontier read.
4. Replay acceptance equals a step-by-step greedy speculative loop on the tiny model.
5. Resumability: kill a run midway, restart it, and get identical final logs.

**Run hygiene**
- One output file per (trace, method, config). Skip it if it already exists. Write to a temp file, then rename.
- Run long jobs under tmux or nohup, with logs in `$PERSIST/logs`.
- Keep `RUNLOG.md` with every command, its start and end time, and cumulative GPU-hours.
- Record pip freeze, GPU name, driver and git commit in `results/ENV.md`.

## 7. Stages and gates

**Stage 0 (CPU)**
- Set up the environments.
- Read the DFlash inference code and write `NOTES.md`. It should cover:
  - which slot predicts which token;
  - how context features are built and appended;
  - where lm_head is applied;
  - the training position weights.
- Build the harness and the unit tests on the tiny model.

**GATE A.** Send me `NOTES.md` and the test results. Wait.

**Stage 1 (GPU, about 1 h).** Generate all 62 traces with vLLM: greedy, thinking on, 16K cap.
- Save token ids, prompt length, finish reason and the loop flag.
- Report the length distribution, the loop count and the GPU time.

**Stage 2 (GPU, about 0.5 h).** Real-vs-replay check.
- Run the actual DFlash speculative generate (greedy) on 3 prompts: 2 AIME and 1 LCB, with `max_new_tokens` 4096.
- Instrument it to log acceptance per round.
- Replay the same outputs.
- **Pass if:** mean τ is within 1%, and at least 98% of rounds are identical. Also report the near-tie rate.

**GATE B.** Send the results. Wait.

**Stage 3 (GPU, about 1.5–2 h).** Dev split.
- Run the baseline and the full grids for 5.2, 5.3 and 5.4, plus the stale-vs-refresh check.
- Pick the best config per method by dev mean τ.
- Freeze the choices in `configs/frozen.yaml`.

**GATE C.** Send me the dev table and the chosen configs. Wait.

**Stage 4 (GPU, about 2 h).** Run the test split with the frozen configs. Run 5.5 only if the rule says GREY.

**Stage 5 (CPU).** Write the report (§8).

## 8. Deliverables

**`results/summary.md`**
- A table of method × {AIME, LCB, mean} with these columns:
  - τ overall;
  - τ for output positions ≥ 8K;
  - R (gain recovered);
  - update time (ms/round);
  - draft forward time (ms/round);
  - adaptation-state memory (MB).
- A decision line: GO / GREY / STOP / INVALID per §1, with the numbers that triggered it.
- Looping traces: the count, and the results with them included.
- Any deviations from this brief, and why.

**Other files**
- `results/plots/tau_vs_position_{aime,lcb}.png`: all methods, 2K bins.
- `results/raw/*.jsonl`: per-round logs.
- `RUNLOG.md`, `NOTES.md`, `results/ENV.md`.

**Sanity anchor (not a gate).** TTS reported a baseline τ of about 3.3 on AIME-24 for this checkpoint with outputs up to 32K. With a 16K cap, expect something similar or slightly higher.

## 9. Stop and ask me if

- the near-tie rate is above 1%, or the real-vs-replay check fails;
- the DFlash indexing does not match §4 (describe it and propose a fix);
- more than 25% of traces are flagged as looping;
- an OOM can't be fixed by reducing batch size;
- projected GPU time goes above 8 h;
- you want to change anything in §1.

## 10. Out of scope

- serving or SGLang integration, CUDA graphs;
- sampling (T > 0);
- other models or drafters;
- kNN memory, tree verification;
- paper writing.

## 11. Amendments (Gate A, before any GPU results)

Approved 2026-10-10, before any GPU run. The decision rule in §1 is unchanged.

1. **RLS prior (correction to §5.3, not a change of method).** The §5.3 recursion does not implement the stated
   objective under forgetting: its prior decays by λ^N, which causes covariance windup (NOTES.md §6). The RLS arm
   therefore uses the **fixed-prior exact solver** of the stated objective (`prior: fixed`).
   - Grid: n0 ∈ {16, 128, 1024} × λ ∈ {0.995, 0.999, 0.9999, **1.0**}.
   - At λ = 1 the O(d²m) recursion solves the same objective with no windup. λ = 1 runs use the recursion
     (`prior: decay`, which equals `fixed` at λ = 1). On one dev trace, check that it matches the fixed solver
     (relative error < 1e-4).
   - Solve in fp32. If a Cholesky ever fails, or the fp32 and fp64 corrected logits differ by > 1e-3 on one dev
     trace, switch to fp64 and log it.
   - Log the solve time per round as its own field (`solve_ms`).
2. **TTS stale vs refresh.** Run refresh at the best stale lr only. If refresh beats stale by > 5% τ there, also run
   refresh at the other two lrs before freezing TTS's config.
3. **LiveCodeBench v6** = `test6.jsonl`, hard, 32 sampled with seed 0.
4. **Thinking mode.** Keep thinking ON with this checkpoint (it matches TTS and gives long outputs). The drafter was
   trained with thinking off, so part of any adaptation gain may come from fixing a constant mode mismatch rather
   than length decay. Therefore:
   - `summary.md` also reports τ and R for the first 2K output tokens, next to the ≥ 8K columns.
   - Control (not part of the decision rule): generate thinking-OFF traces for the 6 dev AIME prompts, replay the
     baseline only, and report its τ.
5. **Prompt templates** from DFlash commit `bb48ef9`: approved.
6. **GPU budget.** At Stage 2, re-project the full GPU budget including the RLS solve time. Stop if it goes above
   8 GPU-hours.
