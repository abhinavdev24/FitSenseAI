FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install huggingface_hub

# Bake base model into image at build time — no runtime download ever.
ARG HF_TOKEN
RUN HF_TOKEN=${HF_TOKEN} python - <<'EOF'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="unsloth/Qwen3-4B-bnb-4bit",
    token=os.environ.get("HF_TOKEN"),
    local_dir="/app/base_model",
    ignore_patterns=["*.md", "*.txt"],
)
print("Base model downloaded to /app/base_model")
EOF

COPY server.py /app/server.py

# Tell server.py to use the baked-in model instead of downloading
ENV FITSENSE_BASE_MODEL_PATH=/app/base_model

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
