# Session 4 artifacts (GitHub release `artifacts-s4`)

Non-git artifacts from session 4: the calibrated drafter checkpoints and the replay traces.
Frozen checkpoints: `ckpt/cal/lr1e-05` (S2+cal, sha256 `cc25cc15…e8ca5d7`) and `ckpt/calfull/lr1e-05`
(cal-full, sha256 `1f3ba7a4…a05b797`); both verified before packaging. `hf_cache/` and `/home/papers_dl`
are not shipped (re-downloadable).

## Release assets

| asset | bytes | sha256 |
|---|---:|---|
| `s2cal_lr1e-05.tar.gz` | 852208371 | `f4c6d28cd89276da61fd7aaa4fd3ec8a2873a0ab67fc1d00ab85f4a6cf99340b` |
| `s2cal_lr3e-06.tar.gz` | 852214038 | `e4b505919814d3427c7c52903d15720d2ac73345ea4abfa97a32059947e657da` |
| `calfull_lr1e-05.tar.gz` | 852208501 | `85cec957aceb802fe556dfd4e99248e312dd1a2ddad74056784fc19684a0f411` |
| `traces.tar.gz` | 889278 | `2f0cdd51b3a6a5b2aabb30a3a3e8487e4d5a03e075d01705bacd78bb63668596` |

`s2cal_*` extract into `ckpt/cal/`, `calfull_*` into `ckpt/calfull/`, `traces.tar.gz` into `traces/`.
Each checkpoint tarball holds `<lr>/model.safetensors` + `<lr>/config.json`. The per-run sidecars
(`summary.json`, `train_log.txt`, which sit one level up and are shared by the lr sweep) are committed in git
under `ckpt_meta/<asset name>/` (and listed below).

## Files (paths relative to repo root)

| file | bytes | sha256 |
|---|---:|---|
| `ckpt/cal/lr1e-05/model.safetensors` | 1074860568 | `cc25cc15a3782b56e2a5be9d1a17bbaf74718be966413447b76a5b728e8ca5d7` |
| `ckpt/cal/lr1e-05/config.json` | 1154 | `539abec79cc772df12c045677d10eb0654564d326c3fcda58d8abfdbaa883878` |
| `ckpt/cal/lr3e-06/model.safetensors` | 1074860568 | `7a0e0fb8d7002feb6f0f51f7c0465733f67157c628bffae5678bd3461021ea0d` |
| `ckpt/cal/lr3e-06/config.json` | 1154 | `539abec79cc772df12c045677d10eb0654564d326c3fcda58d8abfdbaa883878` |
| `ckpt/cal/summary.json` | 68448 | `c81baf823ef76f076ebeea2e4a264cd3c5f6465b3ec45a38e90d5038d4665a85` |
| `ckpt/cal/train_log.txt` | 6682 | `a3cb318b4402d13f0b3c568988b0a0cb8ec4fa295d66e62161b7f931387328ca` |
| `ckpt/calfull/lr1e-05/model.safetensors` | 1074860568 | `1f3ba7a4bddc1f17bade5608eebfb7e6f512b393cb61ff01d39283615a05b797` |
| `ckpt/calfull/lr1e-05/config.json` | 1154 | `539abec79cc772df12c045677d10eb0654564d326c3fcda58d8abfdbaa883878` |
| `ckpt/calfull/summary.json` | 34795 | `63a65cc452d17cb20a4e310898dd928d1e982625c4e51ea3a0e7860270526a35` |
| `ckpt/calfull/train_log.txt` | 4452 | `5c491a760377e83330d1eeaa78f069fd2e9d056c789aa14aaa4fe3a2dab77dae` |
| `traces/traces.jsonl` | 4423219 | `c056ab8b288201e0bdc189deb734f926688351305a2629308789ba44ec5d851a` |
| `traces/traces_cal.jsonl` | 2612885 | `a8e1a2d58384fdd9370ece62d3e62cac89c34b97b76b7285ea491a8865c3da83` |
| `traces/traces_nothink.jsonl` | 359561 | `8c94484c7cc5d39c076ac60f5045754c775b46c11d9e7f81f68864efb1d733a4` |

## Restore

```bash
gh release download artifacts-s4 -D /tmp/art &&
mkdir -p ckpt/cal ckpt/calfull traces &&
tar xzf /tmp/art/s2cal_lr1e-05.tar.gz -C ckpt/cal &&
tar xzf /tmp/art/s2cal_lr3e-06.tar.gz -C ckpt/cal &&
tar xzf /tmp/art/calfull_lr1e-05.tar.gz -C ckpt/calfull &&
tar xzf /tmp/art/traces.tar.gz -C traces
```
Then verify: `sha256sum ckpt/cal/lr*/model.safetensors ckpt/calfull/lr1e-05/model.safetensors traces/*.jsonl`
against the table above.
