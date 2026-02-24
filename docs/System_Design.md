# FitSenseAI System Design (Google Cloud)

Last updated: 2026-02-22

## 1. Scope

This document proposes a concrete system design for FitSenseAI on Google Cloud:

- deploy a backend API for plan generation, coaching, and logging,
- train and serve a student model without Vertex AI (cost-controlled),
- run/scale the offline pipeline (currently local Airflow) in a cloud environment,
- enforce security, privacy, and observability practices suitable for an MVP.

## 2. Design Goals

- Low-latency and predictable-cost online inference via a student model.
- Reproducible and auditable training data and model versions.
- Safety-first behaviors for injury/medical-risk prompts.
- Clear separation between dev/stage/prod environments.

## 3. Proposed GCP Services

- **Cloud Run**: Backend API (stateless, autoscaling).
- **Cloud SQL (PostgreSQL)**: transactional store for user/workout/health data.
- **Cloud Storage (GCS)**: datasets, model artifacts, QA reports, logs exports.
- **Compute Engine (GPU)** / **GCP Batch (GPU)** / **GKE (GPU)**: training and (optionally) GPU-backed inference.
- **Secret Manager**: API keys (teacher provider), DB creds, service credentials.
- **Cloud Build**: CI/CD builds and deploys for backend and pipeline images.
- **Artifact Registry**: container images for backend and batch/pipeline tasks.
- **Self-managed Airflow on Compute Engine** or **Cloud Scheduler + Workflows + Jobs**: orchestration of the offline pipeline in the cloud.
- **Cloud Logging / Monitoring**: logs, metrics, alerting dashboards.

## 4. High-Level Architecture

### 4.1 Online Plane

1. Client calls Backend API on Cloud Run.
2. Backend reads user context from Cloud SQL.
3. Backend calls the student inference service:
   - low-cost path: CPU-quantized model on Cloud Run (if feasible for your chosen model),
   - higher-throughput path: GPU-backed service on GKE/Compute Engine.
4. Backend stores:
   - user logs (workouts + daily signals),
   - plan versions,
   - AI interactions with model version metadata.

### 4.2 Offline Plane

1. Orchestrator triggers the offline pipeline run with a `run_id`.
2. Batch jobs generate synthetic tables, prompts, teacher outputs, and a distillation dataset in GCS.
3. QA jobs compute validation/stats/anomaly/bias slicing reports and store outputs in GCS.
4. Training job fine-tunes the student model from the distillation dataset and writes model artifacts + evaluation results to GCS with version metadata.

## 5. Data Storage Design

### 5.1 Cloud SQL (PostgreSQL)

Use the schema in `database/tables.sql` (normalized relational model), with these MVP notes:

- Use UUID primary keys (already modeled).
- Add indexes for high-volume queries:
  - `workouts(user_id, started_at)`,
  - `ai_interactions(user_id, created_at)`,
  - `workout_sets(workout_exercise_id, set_number)`.
- Store model metadata on each interaction (already modeled as `model_name`; extend to include `model_version` and `request_id` in implementation phase).

### 5.2 GCS Bucket Layout (Recommended)

- `gs://fitsenseai-mlops-{env}/datasets/distillation/<run_id>/...`
- `gs://fitsenseai-mlops-{env}/teacher_outputs/<run_id>/...`
- `gs://fitsenseai-mlops-{env}/reports/phase6/<run_id>/...`
- `gs://fitsenseai-mlops-{env}/models/student/<model_version>/...`

## 6. Training and Serving Design

### 6.1 Training

- Use containerized training for reproducibility on cost-controlled compute.
- Inputs:
  - distillation dataset version (`run_id`),
  - training config (LoRA/QLoRA, epochs, batch size),
  - tokenizer/template choice.
- Outputs:
  - model artifact in GCS,
  - evaluation report in GCS,
  - a simple lineage manifest (dataset `run_id`, git commit SHA, training config, eval report URI).

Recommended MVP execution targets (choose one):

- Compute Engine GPU VM (optionally Spot/Preemptible)
- GCP Batch (GPU) jobs
- GKE Standard with GPU nodes

### 6.2 Serving

Preferred MVP path:

- Single-GPU inference VM (Compute Engine) running **vLLM** (OpenAI-compatible API + streaming), private-only.

Recommended baseline for your constraints (7B max, private endpoint, peak concurrency ~10):

- Region: deploy close to Boston users (start with `us-east4` if available).
- GPU class: prefer 24GB VRAM (L4 or A10). 16GB (T4) is possible but requires stricter limits.
- vLLM settings:
  - `--max-model-len 4096`
  - `--max-num-seqs 2` (start here) or `3` (if stable)
  - enforce `max_output_tokens` in the backend (target ~1024)

Networking:

- Put the inference VM on a private subnet.
- Cloud Run backend reaches the VM via Serverless VPC Access.
- Firewall allow the vLLM port only from the VPC connector subnet.
- Do not expose the model endpoint publicly.

Alternative (when needing more control):

- GKE/Compute Engine running a GPU-backed model server (e.g., vLLM/TGI) behind an internal load balancer; Backend calls it over private network.

## 7. Orchestration Strategy

### 7.1 Short-Term MVP

- Keep local Airflow for development parity (already implemented in `Data-Pipeline/dags/fitsense_pipeline.py`).
- Containerize each pipeline phase script as a Cloud Run Job or a single image with parameters.

### 7.2 Production-Grade Path

Pick one:

- Self-managed Airflow on a small Compute Engine VM: closest to current implementation and cheaper than managed orchestration.
- Cloud Scheduler + Workflows to trigger batch jobs: simpler operational footprint for a small team.

## 8. Security and Privacy

- IAM:
  - separate service accounts for backend, pipeline jobs, and training jobs.
  - least privilege access to Cloud SQL and GCS prefixes.
- Secrets:
  - store teacher provider credentials and DB creds in Secret Manager.
  - mount to Cloud Run / Composer via secret references.
- Network:
  - Cloud Run to Cloud SQL via Serverless VPC Access connector (private connectivity).
  - restrict public access to internal services; expose only API gateway/front door as needed.
- Data handling:
  - synthetic data is default for training in early phases.
  - do not use raw user chat logs for training without explicit opt-in and anonymization (future).

## 9. Observability and Monitoring

### 9.1 Online

- Logs: request_id, user_id (or anonymized), endpoint, latency, model_version.
- Metrics: error rates, p95 latency, rate limits, costs.
- Alerts: sustained error spikes, latency regressions, quota exhaustion.

### 9.2 Offline

- Track per-stage runtime, row counts, validation pass/fail, anomaly severity, bias alerts.
- Store QA JSON reports per `run_id` in GCS.

## 10. Failure Modes and Mitigations (MVP)

- Teacher LLM outage:
  - retry with backoff; allow `mock` provider fallback for development runs.
- Bad teacher outputs contaminating training:
  - enforce post-validation and safety-flag rejection before dataset export.
- Cost spikes:
  - quotas on inference requests; cache common responses where appropriate; use student model by default.

## 11. Open Questions (Confirm Before Implementation)

1. Student model choice and serving constraints (latency vs cost vs quality).
2. Whether the teacher provider will be an external API or a self-hosted open-weights teacher model.
3. Environment separation needs (single project vs per-env projects).
