#!/bin/bash
# setup_vm.sh
# -----------
# One-time setup for the FitSense CPU orchestrator VM.
# Run this once after cloning the repo onto the VM.
# Usage: bash setup_vm.sh

set -e  # exit immediately on any error

echo "========================================="
echo "  FitSense VM Setup"
echo "========================================="

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/5] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    unzip \
    build-essential

# Make python3.11 the default python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
echo "      Python version: $(python3 --version)"

# ── 2. gcloud SDK ─────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Checking gcloud SDK..."
if ! command -v gcloud &> /dev/null; then
    echo "      gcloud not found — installing..."
    curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
    source ~/google-cloud-sdk/path.bash.inc
    gcloud init --skip-diagnostics
else
    echo "      gcloud already installed: $(gcloud --version | head -1)"
fi

# ── 3. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "[3/5] Creating Python virtual environment at ~/fitsense-env..."
python3 -m venv ~/fitsense-env
source ~/fitsense-env/bin/activate

pip install --upgrade pip -q

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "[4/5] Installing Python dependencies..."

# Core ML inference stack — no unsloth/CUDA needed (CPU-only VM)
pip install -q \
    torch==2.2.2 \
    transformers==4.46.3 \
    peft==0.11.1 \
    accelerate==0.33.0 \
    bitsandbytes==0.43.3 \
    sentencepiece \
    protobuf

# Evaluation metrics
pip install -q \
    rouge-score \
    bert-score

# GCP clients
pip install -q \
    google-cloud-storage \
    google-cloud-aiplatform

# Experiment tracking
pip install -q wandb

# Data utilities
pip install -q datasets

echo "      All packages installed."

# ── 5. Shell profile — auto-activate venv ─────────────────────────────────────
echo ""
echo "[5/5] Configuring shell to auto-activate virtualenv..."
ACTIVATE_LINE="source ~/fitsense-env/bin/activate"
if ! grep -qF "$ACTIVATE_LINE" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# FitSense virtualenv" >> ~/.bashrc
    echo "$ACTIVATE_LINE" >> ~/.bashrc
    echo "      Added to ~/.bashrc"
else
    echo "      Already in ~/.bashrc — skipping"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Run: source ~/.bashrc"
echo "  2. Authenticate GCP: gcloud auth application-default login"
echo "  3. Pull your adapter from GCS:"
echo "     gsutil cp -r gs://fitsenseai-model-registry/models/fitsense-qwen3-4b/<run_id>/ \\"
echo "       Model-Pipeline/adapters/qwen3-4b-fitsense/"
echo "  4. Run evaluation:"
echo "     python Model-Pipeline/scripts/evaluate_student.py"
echo "========================================="