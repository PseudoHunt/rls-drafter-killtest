#!/bin/bash
# Records environment details into results/ENV.md (brief §6).
cd "$(dirname "$0")/.."
{
  echo "# ENV"
  echo
  echo "- date: $(date -u +%FT%TZ)"
  echo "- git commit: $(git rev-parse HEAD)"
  echo "- GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "- dflash reference repo: z-lab/dflash $(git -C ext/dflash rev-parse --short HEAD) (pip dflash==0.1.0, model.py identical)"
  echo "- drafter: z-lab/Qwen3-4B-DFlash-b16 @ b74e3a3; target: Qwen/Qwen3-4B"
  for e in env-dflash env-vllm; do
    echo; echo "## $e (python $(/home/envs/$e/bin/python -V 2>&1 | cut -d' ' -f2))"; echo; echo '```'
    VIRTUAL_ENV=/home/envs/$e uv pip freeze 2>/dev/null
    echo '```'
  done
} > results/ENV.md
