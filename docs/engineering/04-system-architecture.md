# FitSenseAI System Architecture

Last updated: 2026-02-22

This is the technical system architecture for MVP deployment on Google Cloud without Vertex AI.

## 1. Components

### 1.1 Online

- Client (web/mobile)
- Backend API (Cloud Run)
- Database (Cloud SQL PostgreSQL)
- Student model inference (Compute Engine VM with 1 GPU, vLLM server, private-only)

### 1.2 Offline

- Synthetic and distillation pipeline (`Data-Pipeline/`, Airflow DAG)
- External teacher API (offline only, called by the pipeline)
- Training job runner (Compute Engine GPU VM or GCP Batch GPU jobs)
- Artifact store (GCS) for datasets, reports, and model versions

## 2. Online Request Path (Streaming)

### 2.1 Plan / Coach Call

1. Client calls backend endpoint.
2. Backend loads user context from Cloud SQL.
3. Backend calls vLLM on the inference VM using OpenAI-compatible API with `stream=true`.
4. Backend streams chunks to the client (SSE or chunked HTTP).
5. Backend stores `ai_interactions` with model name/version and context reference.

## 3. Offline Pipeline Path (Synthetic -> Distillation)

Implemented today under `Data-Pipeline/`:

- synthetic structured generation
- prompt generation with slice tags
- teacher calling with post-validation and safety flags
- distillation dataset creation with deterministic stratified splits
- QA reports: validation, stats, anomaly detection, bias slicing

## 4. Training Path (Distillation -> Student)

MVP training design:

- Input: `distillation_dataset/<run_id>/train.jsonl` in GCS.
- Training: containerized fine-tune on a single GPU.
- Output:
  - model artifact stored in GCS,
  - evaluation JSON report stored in GCS,
  - a model version manifest that ties together dataset `run_id`, git commit SHA, and eval report URI.

## 5. Networking and Security Boundaries

- Inference VM is not publicly accessible.
- Backend reaches inference VM via private network (Serverless VPC Access).
- Firewall rules allow inference port only from the VPC connector subnet.
- Secrets (DB creds, teacher API key) are stored in Secret Manager.

## 6. Capacity and Backpressure

Single GPU inference is capacity-limited. The backend must:

- cap per-request output tokens (target: 1024),
- keep true GPU concurrency low (start at 2-3 in-flight sequences on vLLM),
- queue excess requests (Cloud Tasks recommended) or return `429`.

