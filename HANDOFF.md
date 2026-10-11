# HANDOFF — drafter long-output decay: S2 context + one-time thinking-mode calibration (paused 2026-10-11)

Read this first. Then `results/summary.md` (results, newest first), `RUNLOG.md` (every run, pre-registrations, A30
hours), `NOTES.md` (DFlash internals §1–4, context policies §9–12, batched update + calibration §13) and
`KILL_TEST_BRIEF.md` (original brief + amendments §11–12: replay design and the rules we follow).

## Where things stand (one paragraph)

The original RLS question is closed (STOP). The long-output acceptance decay of the DFlash drafter (Qwen3-4B,
thinking on, 16K) turned out to be a **drafter-context** problem (S2: prompt + last 1024 + 16 far chunks chosen by the
target's attention) plus a **thinking-mode mismatch**. Session 3 fixed the mismatch once, offline: the released DFlash
weights fine-tuned on 59 disjoint thinking-on calibration traces in S2 layout (**S2+cal**, frozen checkpoint sha256
`cc25cc15a3782b56e2a5be9d1a17bbaf74718be966413447b76a5b728e8ca5d7`). On the 49 held-out test traces (replay, A30):
**S2+cal τ 4.408 vs TTS stride 1 4.208: +0.200 [+0.179, +0.222], with zero online update cost** (TTS: 69.6 ms/round of
updates). Refresh-vs-stale fairness check for TTS: refresh − stale = −0.07% (run). "full" (S2+cal + exact batched
pooled TTS stride 10): see summary.md (beats TTS further but fails the ≤ 10% cost criterion: ≈ 11 ms/round). What
calibration buys is not yet split into thinking-mode vs S2-layout adaptation — that is open question (a).

## Rerun S2+cal from the frozen checkpoint (one command)

```bash
cd /home/rls_drafter_killtest && bash scripts/rerun_s2cal.sh
```
It sources `scripts/resume.sh`, verifies the checkpoint hash, replays S2+cal on all 50 test traces into
`results/raw_test_cal_rerun/` (≈ 0.4 A30-h; resumable) and prints τ next to 4.408 (expect MATCH on an A30; other GPU
types differ slightly through bf16 numerics, as in every earlier cross-GPU check).

If `ckpt/` is gone (it is not in git): rebuild with
`python -m rlsdk.calibrate --traces traces/traces_cal.jsonl --lrs 1e-5 --steps 1415 --batch 16 --out ckpt/cal`
(≈ 0.33 A30-h; needs `traces/traces_cal.jsonl`, regenerate with
`$ENV_VLLM/bin/python scripts/gen_traces.py --set cal --out traces/traces_cal.jsonl --max-new-tokens 8192`, ≈ 0.27 A30-h).
A rebuild is the same recipe but not guaranteed bit-identical (vLLM / cuBLAS nondeterminism): it gets a new hash.

## Where things live (only /home survives a JarvisLabs pause)

| what | path |
|---|---|
| repo | `/home/rls_drafter_killtest` (git `github.com/PseudoHunt/rls-drafter-killtest`, private; branch `main`) |
| model weights (`HF_HOME`, offline) | `hf_cache/` (Qwen/Qwen3-4B, z-lab/Qwen3-4B-DFlash-b16; not in git) |
| calibrated drafter | `ckpt/cal/lr1e-05/` (frozen, chosen), `ckpt/cal/lr3e-06/`, `ckpt/cal/summary.json`, `train_log.txt` (not in git) |
| traces (not in git) | `traces/traces.jsonl` (62 dev/test), `traces/traces_cal.jsonl` (60 calibration), `traces/traces_nothink.jsonl` |
| per-round replay logs | `results/raw_test/` (test: baseline, window, S2, C1, C2, TTS, T2 arms, batched stride 10/20), `results/raw_test_cal/` (S2+cal, full), `results/raw_a30*/` (dev / reference traces) |
| venvs | `/home/envs/env-dflash` (torch 2.13, transformers 5.15, dflash 0.1.0), `/home/envs/env-vllm` (vLLM 0.31), `/home/envs/env-jl` (jarvislabs SDK); interpreters in `/home/.local/share/uv/python` |
| resume | `source scripts/resume.sh` → prints `RESUME OK` (sets offline HF env, activates env-dflash, MPS, GPU + dflash import check) |
| JarvisLabs token | `/home/.jarvislabs_token` (mode 600, outside the repo; never print, log or commit). Pause: `$ENV_JL/bin/python scripts/jl_pause.py --pause` (`--check` is read-only); uses `$MACHINE_ID` (instance 534645 this session) |

`/home/rls_vpa` is an old worktree of branch `vpa` (merged); `/home/jl_fs/...` are stale copies / another project.

## Code map

- `rlsdk/replay.py` round simulation (leakage guard, TTS gradient pass); `rlsdk/window.py` context policies (window,
  S2 far chunks, VPA); `rlsdk/adapters.py` arms (Baseline, FarSel = S2, TTS with stride / pool / ctx / batched /
  micro / warm); `rlsdk/batched.py` exact batched pooled update; `rlsdk/calibrate.py` offline calibration;
  `rlsdk/pool.py` resumable workers (`--draft` takes a checkpoint dir); `rlsdk/analyze.py` decision variant (loop
  truncation), bins.
- Reports: `scripts/s3_report.py {step1,screen,heldout,lrpick}` (session 3), `scripts/heldout_report.py` (session 2).
- Checks: `scripts/check_cal_disjoint.py` (calibration set vs the 62 problems; output `results/cal_disjoint.txt`),
  `scripts/batched_check_fp32.py`.
- Tests: `python -m pytest -q tests` on CPU (127 tests, ≈ 3.5 min). Do not run two CPU torch jobs at once (thread
  oversubscription makes both crawl).

## Operational lessons (24 GB A30)

- Never run a TTS-type worker (≈ 13 GB steady) next to another worker: a teacher-forced pass (+8 GB target layers)
  then OOMs. Two light (no-TTS) workers are fine.
- Batched TTS needs `gqa_attention` + micro-batching (both default now); transformers' SDPA path expanded K/V per
  pooled round and OOMed on long-prompt LCB traces.
- `pkill -f <pattern>` / `pgrep -f` from a shell whose command line contains the pattern kills that shell: use PIDs.
- Monitors expire after 30 min; an expired monitor cost 14 idle minutes this session.

## NEXT PHASE (user plan of 2026-10-11, items A–J; cheaper GPU) — NOT pre-registered, NOT run

The user's plan is quoted in the session transcript; its content is summarised here. **Before any GPU run:** (1) the
user confirms the budget (D), (2) pre-register in RUNLOG.md. Fixed: the 49-trace test set, the loop-cut rule, the bins
(0–2K, 2–4K, 4–8K, ≥ 8K), the bootstrap (paired, stratified, B = 10,000, seed 0). Reuse saved S2, TTS stride 1, S2+cal
and full numbers (do not rerun them).

- **A. SharpDraft baseline** (2610.05106): done so far = paper read (NOTES.md §14: Eq. 5, constants k 32, s 5,
  β_max e^0.5, N0 2048 for Qwen, BF16 rule Eq. 18; no public code). To do: `SharpDraft` class in rlsdk/adapters.py
  (query scale after q_norm via Eq. 18, all 5 layers, N = keys actually visible incl. the 16 block slots, same count
  for every arm), CPU tests (g(N0) = 1, cap, scale 1 = baseline bit-exact, q-scaling = logit-scaling vs a reference
  softmax), sanity anchor on the 4 reference traces (τ clearly above baseline, gain growing with position; else stop).
- **B. Arms on the 49 test traces:** 1 SharpDraft (full context); 2 SharpDraft + S2; 3 SharpDraft + S2+cal; 4 cal-full
  (calibrate in the full-context layout, same data/steps/lr, run without window); 5 TTS stride 1 + cal (S2 layout).
- **C. Readings** (fixed by the user; see the transcript): arm 1 within CI of S2+cal → contribution = calibration +
  selection, not the window; arm 3 beats S2+cal and arm 1 (lower CI > 0) → complementary; arm 4 within CI of S2+cal →
  window/S2 layout adds nothing beyond calibration; arm 5 > S2+cal → report it; claim stays "matches or beats TTS at
  zero online cost". *Make "within the CI" precise when pre-registering* (proposal: paired CI of arm − S2+cal contains 0).
- **F. Thinking-mode drafter** `jgeuter/qwen3-4b-dpard-thinking-b16` (3 layers, b16): check it loads in the DFlash
  path (block layout, target feature layers; note differences, do not force); arms baseline, + S2, + TTS stride 1
  lr 1e-5 (cut its TTS first). Reading: its baseline τ ≥ S2+cal → calibration gain is available by retraining on
  thinking data; lead with S2 and the hybrid. Read D-PARD/D-PACE and compare its loss with our calibration KL (NOTES).
- **H. D-PACE slot weights** (2605.18810) as a calibration variant: detached acceptance-weighted weights (Eq. 8,
  α = 0.5; Eq. 8 and App. B.1 differ — use Eq. 8), soft KL kept; same data/steps/lr; one fine-tune; S2 layout on 49.
- **E, G, I. Related work** in NOTES.md: SharpDraft, H-Spec, Strong Drafts Need Compact Memories, Draft-OPD — done
  (§14, from abstracts); still to read: Windowed-MTP (2607.21535), Attention Drift (2605.09992), AngelSpec (2607.25852,
  Table 6: thinking-mode mismatch), D-PACE (2605.18810) / D-PARD (slot weighting).
- **J.** Done (NOTES §14): SharpDraft cites TTS; Full Adaptation = Adam lr 5e-5, stride 5; "no speedup over Frozen"
  holds for Qwen3-8B (1.472× vs 1.489×), **not for Qwen3-4B** (1.886× vs 1.491×).

**Findings that change the plan (raise with the user before pre-registering):**
1. **SharpDraft is ≈ a no-op in the S2 layout.** From our logged visible-key counts (N = visible context + 16):
   AIME β̂ = 1 in 100% of rounds; LCB β̂ > 1 in 59%, mean 1.014, max 1.093. Arms 2 and 3 therefore equal S2 / S2+cal
   exactly on AIME (scale 1 = bit-exact, to be unit-tested) → run them on the 26 LCB traces only (≈ half the cost)
   and state AIME by construction; or drop arm 2. In the full-context layout β̂ is active (mean 1.22–1.25, max 1.44;
   the e^0.5 cap is never reached within 16K).
2. **Budget D underestimates the TTS-type arms.** TTS stride 1 on the 49 test traces took ≈ 3.0 A30-h (session 2,
   phase t1_tts); S2-context TTS has ≈ 56 ms/round of updates vs 70 → arm 5 ≈ 2.5 A30-h (not 1.3); F's TTS arm
   similar (≈ 2.0, 3 layers). cal-full (arm 4) with the same 1,415 × 16 anchors in the full-context layout feeds
   ≈ 2–5× more context rows per anchor (CPU gather + fc/k/v) → ≈ 1.2–1.6 A30-h of fine-tuning (not 0.75) + ≈ 0.7 run.
3. **Cross-GPU reuse.** Reused A30 numbers vs new arms on a different GPU mix bf16 hardware effects into paired
   contrasts whose CIs are ±0.02–0.03. Run a hardware anchor first: S2 and S2+cal on the 4 reference traces on the new
   GPU (≈ 0.1 h) vs the A30 values (S2 4.231, S2+cal 4.540); if they differ by > 0.01 τ, rerun S2+cal (and S2) on the 49.
4. **Memory:** calibration and every TTS arm need ≥ 24 GB with bf16 support (an L4 24 GB works; a 16 GB card or a
   T4 without bf16 does not).

**Revised estimates (A30-h; slower cards scale up — L4 ≈ 1.5–2× A30 time):** A coding + tests ≈ 0.5 (CPU-only box
possible), anchor 0.15; hardware anchor 0.1; arm 1 ≈ 0.7 (baseline-cost replay); arms 2, 3 ≈ 0.25 each (LCB only) or
0.4 each (all 49); arm 4 ≈ 1.9–2.3; arm 5 ≈ 2.5; F: load check ≈ 0.05, baseline ≈ 0.5, + S2 ≈ 0.4, + TTS ≈ 2.0;
H ≈ 0.75 (fine-tune 0.35 + run 0.4). A + B (all five arms, LCB-only 2/3) ≈ 6.6; without arm 5 ≈ 4.1; + F without its
TTS ≈ +1.0; + H ≈ +0.75.

**Earlier open questions, still open:** real (non-replay) loop check on 3 dev traces (≈ 0.5 A30-h + ≈ 1 h coding;
results/loop_check_proposal.md); a second drafter family (EAGLE-3 on Qwen3-8B) and a Llama target (≈ 6–8 GPU-h per
pair on a ≥ 40–48 GB card, plus harness work). cal-full = old (a), TTS + cal = old (b).

## Budget record

Old GPU 2.991 GPU-h; A30 session 2: ≈ 8.98 A30-h; A30 session 3 (this one): ≈ 4.89 A30-h (03:39:37 → pause ≈ 08:33 UTC; cap 5,
extended to ≈ 6 by the user for "full" on test).
