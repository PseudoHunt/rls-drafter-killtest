#!/bin/bash
# Resume after a JarvisLabs pause. Only /home survives: repo, venvs, uv pythons and model weights all live there.
# Usage:  source scripts/resume.sh      (sets env in the current shell; < 2 min, no downloads)
export PERSIST=/home/rls_drafter_killtest
export HF_HOME=$PERSIST/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # never re-download
export PYTHONPATH=$PERSIST
export ENV_DFLASH=/home/envs/env-dflash ENV_VLLM=/home/envs/env-vllm
export PY=$ENV_DFLASH/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
cd $PERSIST || { echo "RESUME FAIL: $PERSIST missing"; return 1 2>/dev/null || exit 1; }
source $ENV_DFLASH/bin/activate

fail=0
for f in $ENV_DFLASH/bin/python $ENV_VLLM/bin/python; do
  [ -x "$(readlink -f $f)" ] || { echo "RESUME FAIL: interpreter for $f missing (uv python under /home/.local/share/uv)"; fail=1; }
done
for m in models--Qwen--Qwen3-4B models--z-lab--Qwen3-4B-DFlash-b16; do
  [ -d $HF_HOME/hub/$m ] || { echo "RESUME FAIL: $HF_HOME/hub/$m missing"; fail=1; }
done
[ -s traces/traces.jsonl ] || { echo "RESUME FAIL: traces/traces.jsonl missing"; fail=1; }

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
  pgrep -f nvidia-cuda-mps-control >/dev/null && echo "MPS daemon up" || { echo "RESUME WARN: MPS daemon not started"; }
  $PY -c "import torch; assert torch.cuda.is_available(); x=torch.randn(1024,1024,device='cuda'); print('torch', torch.__version__, 'GPU ok', float((x@x).abs().mean())>0)" || fail=1
  $PY -c "import transformers, dflash; print('transformers', transformers.__version__, 'dflash ok')" || fail=1
else
  echo "no GPU visible (CPU session)"
  $PY -c "import torch, transformers, dflash; print('torch', torch.__version__, 'transformers', transformers.__version__)" || fail=1
fi
[ $fail = 0 ] && echo "RESUME OK" || echo "RESUME FAILED"
