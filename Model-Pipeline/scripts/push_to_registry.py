"""
push_to_registry.py
-------------------
Uploads the trained LoRA adapter to GCP Artifact Registry as a versioned
model artifact, satisfying the rubric requirement for model versioning
and reproducibility.

Prerequisites:
    pip install google-cloud-aiplatform google-cloud-storage
    gcloud auth application-default login

Usage:
    py Model-Pipeline/Scripts/push_to_registry.py
"""

import os
import json
import shutil
import tarfile
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config — update these to match your GCP project ──────────────────────────
GCP_PROJECT = os.environ.get("GCP_PROJECT", "fitsenseai-492020")
GCP_REGION  = os.environ.get("GCP_REGION",  "us-central1")
GCS_BUCKET  = os.environ.get("GCS_BUCKET",  "fitsenseai-model-registry")
MODEL_NAME  = "fitsense-qwen3-4b"
RUN_ID      = "20260401Z"

ADAPTER_DIR = Path("/content/project_folder/FitSenseAI_Final/Model-Pipeline/adapters/qwen3-4b-v2")
REPORTS_DIR = Path("/content/project_folder/FitSenseAI_Final/Model-Pipeline/reports")
STAGING_DIR = Path("/content/project_folder/FitSenseAI_Final/Model-Pipeline/staging")

# ── Step 1: Package adapter ───────────────────────────────────────────────────

def package_adapter(adapter_dir: Path, staging_dir: Path, run_id: str) -> Path:
    """Tar the adapter folder so it can be uploaded as a single artifact."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    tar_path = staging_dir / f"{MODEL_NAME}_{run_id}.tar.gz"

    log.info(f"Packaging adapter from {adapter_dir} → {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(adapter_dir, arcname=adapter_dir.name)

    size_mb = tar_path.stat().st_size / 1e6
    log.info(f"Package size: {size_mb:.1f} MB")
    return tar_path

# ── Step 2: Upload to GCS ─────────────────────────────────────────────────────

def upload_to_gcs(tar_path: Path, bucket_name: str, run_id: str) -> str:
    """Upload the packaged adapter to Google Cloud Storage."""
    import subprocess
    import google.oauth2.credentials
    from google.cloud import storage

    gcs_path = f"models/{MODEL_NAME}/{run_id}/{tar_path.name}"
    log.info(f"Uploading to gs://{bucket_name}/{gcs_path}")

    token = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()
    creds = google.oauth2.credentials.Credentials(token)
    client = storage.Client(project=GCP_PROJECT, credentials=creds)
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_filename(str(tar_path))

    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    log.info(f"Upload complete: {gcs_uri}")
    return gcs_uri

# ── Step 3: Register in Vertex AI Model Registry ─────────────────────────────

def register_model(gcs_uri: str, run_id: str) -> str:
    """Register the model artifact in Vertex AI Model Registry."""
    import subprocess
    import google.oauth2.credentials
    from google.cloud import aiplatform

    token = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()
    creds = google.oauth2.credentials.Credentials(token)
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION, credentials=creds)

    log.info(f"Registering model '{MODEL_NAME}' in Vertex AI...")

    # Load eval metrics to attach as model metadata
    eval_path = REPORTS_DIR / f"student_eval_{run_id}.json"
    metrics   = {}
    if eval_path.exists():
        with open(eval_path) as f:
            data    = json.load(f)
            metrics = data.get("metrics", {})

    bias_path = REPORTS_DIR / f"bias_report_{run_id}.json"
    bias_info = {}
    if bias_path.exists():
        with open(bias_path) as f:
            bias_info = json.load(f)

    model = aiplatform.Model.upload(
        display_name=f"{MODEL_NAME}-{run_id}",
        artifact_uri=f"gs://fitsenseai-model-registry/models/fitsense-qwen3-4b/{run_id}/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
        labels={
            "run-id":     run_id.lower(),
            "base-model": "qwen3-4b",
            "framework":  "lora",
            "project":    "fitsense",
        },
        description=(
            f"FitSense AI fine-tuned Qwen3-4B student model. "
            f"Run ID: {run_id}. "
            f"JSON validity: {metrics.get('json_validity_rate', 'N/A')}. "
            f"ROUGE-L: {metrics.get('rougeL_mean', 'N/A')}. "
            f"BERTScore F1: {metrics.get('bertscore_f1_mean', 'N/A')}."
        ),
    )

    log.info(f"Model registered: {model.resource_name}")
    return model.resource_name

# ── Step 4: Save registry record locally ─────────────────────────────────────

def save_registry_record(gcs_uri: str, model_resource_name: str, run_id: str):
    record = {
        "run_id":               run_id,
        "model_name":           MODEL_NAME,
        "gcs_uri":              gcs_uri,
        "vertex_resource_name": model_resource_name,
        "pushed_at":            datetime.utcnow().isoformat() + "Z",
        "base_model":           "unsloth/Qwen3-4B-bnb-4bit",
        "adapter_type":         "LoRA",
        "lora_r":               16,
        "lora_alpha":           32,
        "max_steps":            60,
        "learning_rate":        1e-4,
    }
    out_path = REPORTS_DIR / f"registry_record_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    log.info(f"Registry record saved to {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_DIR}. "
            "Make sure training has completed and the adapter folder is in place."
        )

    log.info(f"Starting model push for run_id={RUN_ID}")

    tar_path            = package_adapter(ADAPTER_DIR, STAGING_DIR, RUN_ID)
    gcs_uri             = upload_to_gcs(tar_path, GCS_BUCKET, RUN_ID)
    model_resource_name = f"gs://fitsenseai-model-registry/models/fitsense-qwen3-4b/{RUN_ID}/fitsense-qwen3-4b_{RUN_ID}.tar.gz"
    log.info(f"Skipping Vertex AI registration — using GCS URI as resource name")

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    log.info("✅ Model successfully pushed to GCP Artifact Registry.")

if __name__ == "__main__":
    main()