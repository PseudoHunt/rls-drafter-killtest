"""Test 5: kill a run midway, restart, and get identical final logs (timing fields excluded)."""
import glob
import json
import os
import signal
import subprocess
import sys
import time

import torch
import yaml
from dflash.model import dflash_generate

from rlsdk.data import loop_flag
from rlsdk.tiny import tiny_drafter, tiny_target

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMING = {"draft_ms", "update_ms", "wall_s"}
PLAN = [
    {"method": "baseline"},
    {"method": "bias", "config": {"rho": 0.9}},
    {"method": "rls", "config": {"lam": 0.99, "n0": 16, "sigma2": 1.0}},
    {"method": "tts", "config": {"lr": 1e-3}},
]


def _write_traces(path, n=3, new=200):
    t = tiny_target()
    d = tiny_drafter(t)
    with open(path, "w") as f:
        for i in range(n):
            g = torch.Generator().manual_seed(i)
            prompt = torch.randint(0, 7, (1, 20), generator=g)
            out = dflash_generate(d, t, prompt, new, None)[0].tolist()
            f.write(json.dumps({"trace_id": f"tiny{i}", "dataset": "aime" if i % 2 else "lcb", "split": "dev",
                                "prompt_ids": out[:20], "prompt_len": 20, "output_ids": out[20:],
                                "finish_reason": "length", "loop": loop_flag(out[20:])}) + "\n")


def _cmd(traces, plan, out):
    return [sys.executable, "-m", "rlsdk.run", "--tiny", "--device", "cpu", "--traces", traces,
            "--plan", plan, "--out", out, "--max-new-tokens", "200"]


def _load(out):
    res = {}
    for p in sorted(glob.glob(os.path.join(out, "*.jsonl"))):
        rows = [{k: v for k, v in json.loads(line).items() if k not in TIMING} for line in open(p)]
        res[os.path.basename(p)] = rows
    return res


def test_kill_and_resume(tmp_path):
    traces, plan = str(tmp_path / "traces.jsonl"), str(tmp_path / "plan.yaml")
    _write_traces(traces)
    yaml.safe_dump(PLAN, open(plan, "w"))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ROOT}

    clean = str(tmp_path / "clean")
    subprocess.run(_cmd(traces, plan, clean), check=True, env=env, cwd=ROOT, capture_output=True)

    resumed = str(tmp_path / "resumed")
    proc = subprocess.Popen(_cmd(traces, plan, resumed), env={**env, "RLSDK_SLEEP_PER_ROUND": "0.003"}, cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline and len(glob.glob(os.path.join(resumed, "*.jsonl"))) < 5:
        time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    n_done = len(glob.glob(os.path.join(resumed, "*.jsonl")))
    assert 0 < n_done < 3 * len(PLAN), n_done  # really killed midway

    subprocess.run(_cmd(traces, plan, resumed), check=True, env=env, cwd=ROOT, capture_output=True)
    a, b = _load(clean), _load(resumed)
    assert len(a) == 3 * len(PLAN)
    assert a == b
    assert not glob.glob(os.path.join(resumed, "*.tmp"))
