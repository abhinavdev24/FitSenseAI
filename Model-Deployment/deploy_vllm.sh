#!/usr/bin/env bash
# =============================================================================
# FitSenseAI — vLLM Deployment Script
# =============================================================================
# Platform  : Google Compute Engine GPU VM (Deep Learning VM image)
# Model     : Qwen/Qwen3-4B (base) + LoRA adapter from HuggingFace
#             abhinav241998/qwen3-4b-fitsense-qlora
# Serves as : OpenAI-compatible /v1/chat/completions endpoint
#
# Prerequisites (on the GCE VM):
#   - NVIDIA GPU with >= 40 GB VRAM (recommended: A100 40 GB)
#   - CUDA 12.1+ driver (included in Deep Learning VM images)
#   - Python 3.11+ in PATH
#   - uv package manager: pip install uv
#
# Set as the instance startup script via GCE instance metadata so that it
# runs automatically every time the VM boots:
#   gcloud compute instances add-metadata INSTANCE_NAME \
#     --metadata-from-file startup-script=Model-Deployment/deploy_vllm.sh
#
# The server starts on PORT (default 8000). After opening a firewall rule,
# it is accessible at:
#   http://<VM-EXTERNAL-IP>:8000
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR="/opt/fitsense"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# ---------------------------------------------------------------------------
# Virtual environment (created once, reused on VM restarts)
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
  echo "[deploy] Installing uv..."
  pip install -q uv

  echo "[deploy] Creating .venv with uv..."
  uv venv .venv
  source .venv/bin/activate

  # vllm — the inference server
  # hf_transfer — faster HuggingFace model download
  uv pip install vllm hf_transfer
else
  echo "[deploy] Reusing existing .venv"
  source .venv/bin/activate
fi

# ---------------------------------------------------------------------------
# Port configuration
# ---------------------------------------------------------------------------
PORT=${PORT:-8000}
echo "[deploy] Serving on port ${PORT}"

# ---------------------------------------------------------------------------
# API key — load from .env if present, otherwise use hardcoded default
# ---------------------------------------------------------------------------
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

API_KEY="${API_KEY:?API_KEY is not set. Add API_KEY=<your-key> to .env or export it before running.}"

if [ -z "${API_KEY}" ]; then
  API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
  echo "API_KEY=${API_KEY}" >> .env
  echo "[deploy] Generated new API key and saved to .env"
fi

echo "[deploy] API Key: ${API_KEY}"

# ---------------------------------------------------------------------------
# PyTorch / vLLM tuning flags
# ---------------------------------------------------------------------------
# expandable_segments: reduces GPU memory fragmentation for variable-length
# sequences (critical for long workout plan outputs, up to ~4K tokens)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Suppress vLLM telemetry
export VLLM_NO_USAGE_STATS=1

# Faster HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---------------------------------------------------------------------------
# Start vLLM server
# ---------------------------------------------------------------------------
echo "[deploy] Starting vLLM server..."
echo "[deploy] Base model : Qwen/Qwen3-4B"
echo "[deploy] LoRA adapter: abhinav241998/qwen3-4b-fitsense-qlora (alias: fitsense)"

vllm serve Qwen/Qwen3-4B \
  --api-key "${API_KEY:?API_KEY env var is required}" \
  --port "${PORT}" \
  \
  `# LoRA adapter configuration` \
  --enable-lora \
  --lora-modules fitsense=abhinav241998/qwen3-4b-fitsense-qlora \
  --max-lora-rank 8 \
  \
  `# Memory and sequence length` \
  --max-model-len 17000 \
  --gpu-memory-utilization 0.85 \
  \
  `# Data type — bfloat16 is stable on Ampere+ GPUs and saves memory vs float32` \
  --dtype bfloat16
