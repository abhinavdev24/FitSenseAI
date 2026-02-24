# FitSenseAI MVP Tech Doc

Last updated: 2026-02-22

## 1. Purpose

This document is the engineering runbook for the FitSenseAI MVP, covering:

- what is implemented now (offline pipeline in `Data-Pipeline/`),
- what needs to be implemented next (student training + backend + deployment on Google Cloud),
- practical contracts and operational notes for reproducibility and iteration.

## 2. Current Implementation Snapshot (Repo Reality)

Implemented today:

- Synthetic-data MLOps pipeline (Phases 1-6) in `Data-Pipeline/scripts/`.
- Airflow DAG orchestration in `Data-Pipeline/dags/fitsense_pipeline.py`.
- Deterministic configuration in `Data-Pipeline/params.yaml`.
- Unit/integration tests in `Data-Pipeline/tests/`.
- Relational schema in `database/tables.sql`.

Not implemented yet (planned):

- Student fine-tuning pipeline (cost-controlled training job + evaluation harness).
- Backend API for onboarding, plan CRUD, logging, and inference integration.
- Production deployment on GCP (Cloud Run, Cloud SQL, model serving, Secret Manager).

## 3. Local Development (Offline Pipeline)

### 3.1 Python Environment

- Python virtualenv and deps:
  - install: `pip install -r Data-Pipeline/requirements.txt`

### 3.2 Run Script-by-Script

From repo root:

- `python Data-Pipeline/scripts/bootstrap_phase1.py`
- `python Data-Pipeline/scripts/generate_synthetic_profiles.py`
- `python Data-Pipeline/scripts/generate_synthetic_workouts.py`
- `python Data-Pipeline/scripts/generate_synthetic_queries.py`
- `python Data-Pipeline/scripts/call_teacher_llm.py`
- `python Data-Pipeline/scripts/build_distillation_dataset.py`
- `python Data-Pipeline/scripts/validate_data.py`
- `python Data-Pipeline/scripts/compute_stats.py`
- `python Data-Pipeline/scripts/detect_anomalies.py`
- `python Data-Pipeline/scripts/bias_slicing.py`

Artifacts appear under `Data-Pipeline/data/` and logs under `Data-Pipeline/logs/`.

### 3.3 Run via Airflow (Local)

The DAG calls the same scripts using `BashOperator`. Example (from `Data-Pipeline/README.md`):

- set `AIRFLOW_HOME` and `AIRFLOW__CORE__DAGS_FOLDER`
- set `FITSENSE_PYTHON_BIN` to your Python interpreter
- `airflow dags test fitsense_pipeline <date>`

## 4. Teacher Provider Configuration (Offline)

Teacher calling is configured in `Data-Pipeline/params.yaml` under `phase4.teacher_llm`.

MVP options:

- `provider: mock` for deterministic offline runs.
- `provider: openai_compatible` for a compatible endpoint and key supplied via the env var named by `api_key_env`.

Operational notes:

- Store provider secrets in GCP Secret Manager for cloud runs.
- Log request_id, latency, retries, and safety flags for auditing.

## 5. Student Training on Google Cloud (Planned MVP Implementation)

### 5.1 Training Input Contract

- Training-ready dataset: `distillation_dataset/<run_id>/train.jsonl` (and `val.jsonl`, `test.jsonl`).
- Each record conceptually contains:
  - `instruction` (prompt text),
  - `context` (metadata: prompt_type, slice_tags, constraints),
  - `response` (teacher output).

### 5.2 Training Job on Cost-Controlled Compute (Recommended)

Implement a containerized training entrypoint that:

1. Downloads the chosen dataset `run_id` from GCS.
2. Fine-tunes a student model (LoRA/QLoRA recommended for MVP iteration speed).
3. Evaluates on:
   - slice-tagged prompts,
   - curated safety prompts,
   - held-out `test.jsonl`.
4. Writes:
   - model artifacts to GCS,
   - evaluation JSON report to GCS,
   - and registers the model via a simple version manifest (GCS URI + git commit SHA + dataset `run_id` + eval report URI).

Recommended execution targets (choose one):

- Compute Engine GPU VM (optionally Spot/Preemptible)
- GCP Batch (GPU) jobs
- GKE Standard with GPU nodes

## 6. Backend Deployment on Google Cloud (Planned MVP Implementation)

### 6.1 Backend Service Responsibilities

- Auth and user management (MVP can start with minimal auth, but isolate user data).
- CRUD: plans, workouts, sets, daily logs.
- Inference routing:
  - call student endpoint by default,
  - optional teacher fallback for admin/testing only.
- Interaction logging:
  - store prompts/responses (or references) with model version and timing.

### 6.2 Recommended GCP Building Blocks

- Cloud Run: backend API
- Cloud SQL (Postgres): core transactional DB
- Secret Manager: DB creds + teacher provider key
- Student inference:
  - Cloud Run (CPU-quantized) for smaller models
  - GKE/Compute Engine (GPU) for larger models or higher throughput
- Cloud Logging/Monitoring: telemetry and alerting

## 6.3 Recommended Serving Choice (Your Setup)

Given:

- student model up to 7B
- streaming responses
- private-only model endpoint called by the backend
- peak concurrency around 10

Recommend:

- **Compute Engine VM (1 GPU) running vLLM** as an internal OpenAI-compatible inference endpoint.
- **Cloud Run backend** enforces request limits and handles backpressure/queueing.

Operational notes:

- True GPU concurrency should be kept low (start at 2-3 in-flight sequences) and queue the rest.
- Enforce `max_output_tokens` and context limits in the backend to protect latency and cost.

## 7. CI/CD and Environments (Recommended)

Minimum environments:

- `dev`: rapid iteration, lower quotas
- `stage`: integration testing with stable model versions
- `prod`: user-facing

Recommended CI/CD milestones:

- build/push images to Artifact Registry,
- deploy backend to Cloud Run,
- deploy/upgrade pipeline jobs and orchestrator,
- register promoted models and update serving endpoint.

## 8. Acceptance Checklist (MVP)

- Offline pipeline runs end-to-end and produces:
  - distillation dataset splits,
  - validation/stats/anomaly/bias reports.
- Student model can be trained on GCP compute (Compute Engine/GCP Batch/GKE) using a selected `run_id` dataset.
- Backend can:
  - generate a plan using the student model,
  - log workouts and daily signals,
  - store AI interactions with model version metadata.
- Basic monitoring dashboards exist (latency/error/cost) for both offline and online components.
