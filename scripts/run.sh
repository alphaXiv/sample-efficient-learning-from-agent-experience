#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/tmp/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "ORX_EVIDENCE_START"
date -u '+started_at=%Y-%m-%dT%H:%M:%SZ'
echo "backend=kubernetes"
echo "gpu_model=NVIDIA RTX PRO 6000 Blackwell"
echo "allocated_gpus=4"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apt-get update -qq
apt-get install -y -qq openjdk-17-jre-headless >/dev/null
python -m pip install -q -r requirements.txt

python - <<'PY'
from huggingface_hub import snapshot_download
import json
cfg = json.load(open("config.json"))
snapshot_download(cfg["model"])
PY

python scripts/run_reproduction.py
date -u '+finished_at=%Y-%m-%dT%H:%M:%SZ'
echo "ORX_EVIDENCE_END"
