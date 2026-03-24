# ============================================================
# Stage 1: Training image
# ============================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS training

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install Python dependencies first (for layer caching)
COPY Model-Pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Model-Pipeline code
COPY Model-Pipeline/ /app/Model-Pipeline/

# Copy data pipeline output (training data) - mount at runtime if needed
# Example: docker run -v /path/to/Data-Pipeline/data/raw:/app/Data-Pipeline/data/raw ...

# Environment variables (override at runtime)
ENV WANDB_API_KEY=""
ENV MODEL_NAME="Qwen/Qwen3-8B"
ENV CONFIG_PATH="Model-Pipeline/config/training_config.yaml"
ENV OUTPUT_DIR="Model-Pipeline/outputs"

# Default entrypoint: training
ENTRYPOINT ["python", "Model-Pipeline/scripts/train.py"]
CMD ["--config", "Model-Pipeline/config/training_config.yaml"]

# ============================================================
# Stage 2: Lightweight evaluation/inference image
# ============================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS inference

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY Model-Pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only scripts needed for evaluation/inference
COPY Model-Pipeline/scripts/ /app/Model-Pipeline/scripts/
COPY Model-Pipeline/config/ /app/Model-Pipeline/config/

ENV MODEL_NAME="Qwen/Qwen3-8B"
ENV CONFIG_PATH="Model-Pipeline/config/training_config.yaml"

ENTRYPOINT ["python", "Model-Pipeline/scripts/evaluate.py"]
CMD ["--config", "Model-Pipeline/config/training_config.yaml"]
