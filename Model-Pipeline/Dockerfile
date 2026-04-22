# =============================================================================
# FitSenseAI — Model Pipeline
# Multi-stage build with two targets:
#
#   training  — CUDA devel stack; runs train.py, hparam_search.py
#   inference — CUDA runtime stack; runs evaluate.py, bias_detection.py,
#               sensitivity.py, select_model.py, push_to_registry.py
#
# Build:
#   docker build --target training  -t fitsense-train -f Model-Pipeline/Dockerfile .
#   docker build --target inference -t fitsense-eval  -f Model-Pipeline/Dockerfile .
#
# Run training:
#   docker run --gpus all \
#     -v $(pwd)/Model-Pipeline/data:/app/Model-Pipeline/data \
#     -v $(pwd)/Model-Pipeline/outputs:/app/Model-Pipeline/outputs \
#     -e WANDB_API_KEY=$WANDB_API_KEY \
#     fitsense-train \
#     --config Model-Pipeline/config/training_config.yaml
# =============================================================================

# -----------------------------------------------------------------------------
# Stage: training
# Full CUDA devel image — required for bitsandbytes + unsloth compilation.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS training

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

# Install Python deps before copying source so this layer is cached on code-only changes.
COPY Model-Pipeline/requirements.txt Model-Pipeline/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r Model-Pipeline/requirements.txt

# Copy project files needed for training.
COPY Model-Pipeline/ Model-Pipeline/
COPY Data-Pipeline/data/ Data-Pipeline/data/

ENTRYPOINT ["python", "Model-Pipeline/scripts/train.py"]
CMD ["--config", "Model-Pipeline/config/training_config.yaml"]


# -----------------------------------------------------------------------------
# Stage: inference
# Lighter CUDA runtime image — evaluation, bias detection, registry push.
# bitsandbytes and unsloth are still required to load the quantized adapter.
# Excludes training-only packages: trl, optuna.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS inference

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

# Install inference-only deps: exclude trl (SFTTrainer) and optuna (hparam search).
COPY Model-Pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    grep -vE "^(trl|optuna)[>=<]" /tmp/requirements.txt | \
    pip install -r /dev/stdin

# Copy project files.
COPY Model-Pipeline/ Model-Pipeline/
COPY Data-Pipeline/data/ Data-Pipeline/data/

ENTRYPOINT ["python", "Model-Pipeline/scripts/evaluate.py"]
CMD ["--help"]
