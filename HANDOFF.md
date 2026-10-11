# HANDOFF — drafter long-output decay: S2 context + one-time thinking-mode calibration (paused 2026-10-11, session 4)

Read this first. Then `results/summary.md` (results, newest first), `RUNLOG.md` (every run, pre-registrations, A30
hours), `NOTES.md` (DFlash internals §1–4, context policies §9–12, batched update + calibration §13, SharpDraft /
related work §14–15) and `KILL_TEST_BRIEF.md` (original brief + amendments §11–12: replay design and rules).

**Non-git artifacts are in GitHub release `artifacts-s4`** (checkpoints `ckpt/cal/lr1e-05`, `ckpt/cal/lr3e-06`,
`ckpt/calfull/lr1e-05` and `traces/*.jsonl`); restore with:
```bash
gh release download artifacts-s4 -D /tmp/art &&
mkdir -p ckpt/cal ckpt/calfull traces &&
tar xzf /tmp/art/s2cal_lr1e-05.tar.gz -C ckpt/cal &&
tar xzf /tmp/art/s2cal_lr3e-06.tar.gz -C ckpt/cal &&
tar xzf /tmp/art/calfull_lr1e-05.tar.gz -C ckpt/calfull &&
tar xzf /tmp/art/traces.tar.gz -C traces
```
then verify hashes against `results/ARTIFACTS.md`. Checkpoint sidecars (config/summary/train_log) are also in git
under `ckpt_meta/`. `hf_cache/` and `/home/papers_dl` are not shipped and are re-downloadable.

## Where things stand (one paragraph)

The original RLS question is closed (STOP). The long-output acceptance decay of the DFlash drafter (Qwen3-4B,
thinking on, 16K) has two causes, both fixed without online updates: a **thinking-mode mismatch** (fixed once,
offline, by calibrating the released weights on 59 disjoint thinking-on traces) and a **long-context decay inside
the drafter** (fixed by the S2 context: prompt + last 1024 + 16 far 32-token chunks chosen by the target's
attention). **S2+cal beats TTS stride 1 at zero online cost: τ 4.408 vs 4.208, +0.200 [+0.179, +0.222]** (49
held-out test traces, replay, A30; frozen checkpoint sha256 `cc25cc15a3782b56e2a5be9d1a17bbaf74718be966413447b76a5b728e8ca5d7`).
Session 4 closed the open ablations: **cal-full** (calibration without the S2 layout) 4.166 → S2 adds +0.242
[+0.206, +0.272] on top of calibration, almost all at ≥ 4K; calibration alone is the largest single factor (+1.066
over the released baseline 3.100) and nearly matches TTS (−0.042 [−0.077, −0.001]). **SharpDraft** (training-free query
scaling, arXiv 2610.05106) helps the full-context drafter (+0.544) but is far below S2 (−0.425) and S2+cal (−0.764);
on top of S2+cal it adds +0.002 (rule for "complementary" met, size negligible: S2 keeps the visible-key count near
N0, so its scale is 1 on every AIME round). A public **thinking-mode-trained drafter** (jgeuter D-PARD, 3 layers) is
below S2+cal by 0.820. "full" (S2+cal + batched pooled TTS stride 10) is the cost–τ trade-off point: +0.140 over S2+cal
for 11.4 ms/round (6.1× cheaper than TTS), failing the pre-registered ≤ 10% cost rule.

## Rerun S2+cal from the frozen checkpoint (one command)

```bash
cd /home/rls_drafter_killtest && bash scripts/rerun_s2cal.sh
```
Sources `scripts/resume.sh`, verifies the checkpoint hash, replays S2+cal on all 50 test traces into
`results/raw_test_cal_rerun/` (≈ 0.4 A30-h; resumable), prints τ next to 4.408 (expect MATCH on an A30).
If `ckpt/` is gone (not in git): `python -m rlsdk.calibrate --traces traces/traces_cal.jsonl --lrs 1e-5 --steps 1415
--batch 16 --out ckpt/cal` (≈ 0.33 A30-h); a rebuild is the same recipe but gets a new hash (vLLM / cuBLAS
nondeterminism). cal-full: add `--layout full --micro 4 --out ckpt/calfull` (≈ 0.56 A30-h; sha256 of the session-4
checkpoint `1f3ba7a4bddc1f17bade5608eebfb7e6f512b393cb61ff01d39283615a05b797`).

## Where things live (only /home survives a pause)

| what | path |
|---|---|
| repo | `/home/rls_drafter_killtest` (git `github.com/PseudoHunt/rls-drafter-killtest`, public; branch `main`) |
| model weights (`HF_HOME`, offline) | `hf_cache/` (Qwen/Qwen3-4B, z-lab/Qwen3-4B-DFlash-b16, jgeuter/qwen3-4b-dpard-thinking-b16 @ 78baf29; not in git) |
| calibrated drafters | `ckpt/cal/lr1e-05/` (S2+cal, frozen), `ckpt/cal/lr3e-06/`, `ckpt/calfull/lr1e-05/` (cal-full); `summary.json`, `train_log.txt` each (not in git) |
| traces (not in git) | `traces/traces.jsonl` (62 dev/test), `traces/traces_cal.jsonl` (60 calibration), `traces/traces_nothink.jsonl` |
| per-round replay logs (in git) | `results/raw_test/` (baseline, window, S2, C1, C2, TTS, T2 arms, batched), `results/raw_test_cal/` (S2+cal, full), `results/raw_test_s4/` (SharpDraft), `results/raw_test_cal_s4/` (SD+S2+cal, LCB), `results/raw_test_think/` (thinking drafter), `results/raw_test_calfull/` (cal-full), `results/raw_a30*/` (dev / reference traces) |
| papers read (not in git) | `/home/papers_dl/` (SharpDraft PDF sha256 26c7a6ac…592a, AngelSpec PDF, abstracts) |
| venvs | `/home/envs/env-dflash` (torch 2.13, transformers 5.15, dflash 0.1.0), `/home/envs/env-vllm` (vLLM 0.31); interpreters in `/home/.local/share/uv/python` |
| resume | `source scripts/resume.sh` → `RESUME OK` (offline HF env, env-dflash, MPS, GPU + dflash check; warns if `gh` is missing) |
| git push | credential helper = `gh` (auth in `/home/.config/gh`, survives); the binary in `/usr/bin/gh` does **not** survive a pause — reinstall gh 2.62.0 as `resume.sh` prints (checksum-verified) |
| pause | Pause the instance from the provider's console. |

## Code map

- `rlsdk/replay.py` round simulation (leakage guard, TTS gradient pass, `scaled_queries` = SharpDraft Eq. 18 hook);
  `rlsdk/window.py` context policies (window, S2 far chunks, VPA); `rlsdk/adapters.py` arms (Baseline, FarSel = S2,
  SharpDraft + `sharp_scale`, TTS with stride / pool / ctx / batched / micro / warm); `rlsdk/batched.py` exact batched
  pooled update; `rlsdk/calibrate.py` offline calibration (`--layout s2|full`, `--micro`); `rlsdk/pool.py` resumable
  workers (`--draft` takes a checkpoint dir or HF id); `rlsdk/analyze.py` decision variant (loop truncation), bins.
- Reports: `scripts/s4_report.py {anchor,item1,item2,item3}` (session 4), `scripts/s3_report.py` (session 3),
  `scripts/heldout_report.py` (session 2; `Boot` = the bootstrap used everywhere).
- Launchers (session 4): `scripts/run_s4_anchor.sh`, `run_s4_item1.sh`, `run_s4_item2.sh`, `run_s4_item3.sh`; plans
  `configs/plan_s4_*.yaml`.
- Tests: `python -m pytest -q tests` on CPU (143 tests, ≈ 4 min; `tests/test_sharpdraft.py`, `tests/test_calibrate.py`
  new/extended in session 4). Do not run two CPU torch jobs at once.

## Operational lessons (24 GB A30)

- Never run a TTS-type worker (≈ 13 GB steady) next to another worker. Two light (no-TTS) workers are fine (≈ 11–13 GB
  each); a light replay of the 49 test traces with 2 workers takes ≈ 0.25–0.4 A30-h.
- cal-full training peaks at ≈ 23 GB with micro 4 (16 anchors as 4 length-sorted sub-batches); use `--micro 8` if a
  different card has less headroom.
- `pkill -f` / `pgrep -f` from a shell whose command line contains the pattern kills that shell: use PIDs.
- Wait on long runs with background until-loops on the launcher's END marker (no foreground sleeps).
- A chain `a && b && nohup c &` backgrounds the whole chain — run commit/push and the next launch as separate commands.

## Open questions (not run; estimates in A30-h)

a. **TTS stride 1 lr 1e-5 on top of S2+cal** (user item 4; S2 layout): the strongest online baseline. ≈ 2.5 on the 49
   traces (one TTS worker alone on the GPU). Reading fixed by the user: if it beats S2+cal, report it; the claim stays
   "matches or beats TTS at zero online cost".
b. **S2+cal with D-PACE slot weights** (user item 5; arXiv 2605.18810 Eq. 8, α = 0.5, detached; Eq. 8 and App. B.1
   differ — use Eq. 8; soft KL kept; same data, steps, lr): ≈ 0.35 fine-tune + 0.4 run.
c. **Thinking drafter + TTS stride 1** (user item 6): ≈ 2.0 (3 layers).
d. **SharpDraft + cal-full** (not pre-registered; cheap, ≈ 0.4): whether query scaling recovers part of cal-full's
   ≥ 8K decay (cal-full ≥ 8K 4.187 vs S2+cal 4.756).
e. Real (non-replay) loop check on 3 dev traces (≈ 0.5 + ≈ 1 h integration coding; results/loop_check_proposal.md).
f. A second drafter family (EAGLE-3 on Qwen3-8B) and a Llama target (≈ 6–8 GPU-h per pair on a ≥ 40–48 GB card).

## Budget record

Old GPU 2.991 GPU-h; A30 session 2 ≈ 8.98 A30-h; session 3 ≈ 4.89 A30-h; **session 4 (08:36:37 → pause ≈ 10:45 UTC)
≈ 2.15 A30-h** (cap 6; details and the exact end time in RUNLOG.md).
