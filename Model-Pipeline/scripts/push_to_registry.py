"""
push_to_registry.py
-------------------
Uploads the trained LoRA adapter to GCS as a versioned model artifact,
satisfying the rubric requirement for model versioning and reproducibility.

Prerequisites:
    pip install google-cloud-aiplatform google-cloud-storage
    gcloud auth application-default login

Usage:
    py Model-Pipeline/scripts/push_to_registry.py
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

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT = os.environ.get("GCP_PROJECT", "mlops-gcp-lab-cloudrunner")  # your project
GCP_REGION  = os.environ.get("GCP_REGION",  "us-central1")
GCS_BUCKET  = os.environ.get("GCS_BUCKET",  "fitsense-adapter-store")     # your bucket
MODEL_NAME  = "fitsense-qwen3-4b"
RUN_ID      = "20260403Z"

ADAPTER_DIR = Path("Model-Pipeline/adapters/qwen3-4b-fitsense")           # your adapter path
REPORTS_DIR = Path("Model-Pipeline/reports")
STAGING_DIR = Path("Model-Pipeline/staging")


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
    from google.cloud import storage
    from google.auth import default as google_auth_default

    gcs_path = f"models/{MODEL_NAME}/{run_id}/{tar_path.name}"
    log.info(f"Uploading to gs://{bucket_name}/{gcs_path}")

    # Use Application Default Credentials — no gcloud subprocess needed
    creds, _ = google_auth_default()
    client = storage.Client(project=GCP_PROJECT, credentials=creds)
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_filename(str(tar_path))

    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    log.info(f"Upload complete: {gcs_uri}")
    return gcs_uri


# ── Step 3: Save registry record locally ─────────────────────────────────────

def save_registry_record(gcs_uri: str, run_id: str):
    """Load eval metrics if available and save a registry record JSON."""
    # Load eval metrics if report exists
    metrics   = {}
    eval_path = REPORTS_DIR / f"student_eval_{run_id}.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data    = json.load(f)
            metrics = data.get("metrics", {})
        log.info(f"Loaded eval metrics from {eval_path}")
    else:
        log.warning(f"Eval report not found at {eval_path} — metrics will be N/A")

    # Load bias report if exists
    bias_info = {}
    bias_path = REPORTS_DIR / f"bias_report_{run_id}.json"
    if bias_path.exists():
        with open(bias_path) as f:
            bias_info = json.load(f)
        log.info(f"Loaded bias report from {bias_path}")
    else:
        log.warning(f"Bias report not found at {bias_path} — bias info will be empty")

    record = {
        "run_id":               run_id,
        "model_name":           MODEL_NAME,
        "gcs_uri":              gcs_uri,
        "pushed_at":            datetime.utcnow().isoformat() + "Z",
        "base_model":           "Qwen/Qwen3-4B",
        "adapter_type":         "LoRA",
        "lora_r":               16,
        "lora_alpha":           32,
        "max_steps":            60,
        "learning_rate":        1e-4,
        "metrics": {
            "json_validity_rate": metrics.get("json_validity_rate", "N/A"),
            "rougeL_mean":        metrics.get("rougeL_mean", "N/A"),
            "bertscore_f1_mean":  metrics.get("bertscore_f1_mean", "N/A"),
        },
        "bias_summary":         bias_info.get("disparities", {}),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"registry_record_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    log.info(f"Registry record saved to {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_DIR}. "
            "Make sure the adapter folder is in place before running this script."
        )

    log.info(f"Starting model push for run_id={RUN_ID}")

    # Package and upload adapter
    tar_path = package_adapter(ADAPTER_DIR, STAGING_DIR, RUN_ID)
    gcs_uri  = upload_to_gcs(tar_path, GCS_BUCKET, RUN_ID)

    # Save registry record with metrics
    record_path = save_registry_record(gcs_uri, RUN_ID)

    # Clean up staging
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

    log.info(f"GCS URI:         {gcs_uri}")
    log.info(f"Registry record: {record_path}")
    log.info("✅ Model successfully pushed to GCS.")


if __name__ == "__main__":
    main()
