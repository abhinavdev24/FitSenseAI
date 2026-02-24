# FitSenseAI Architecture

Last updated: 2026-02-22

## 1. Architecture Summary

FitSenseAI has two major planes:

- **Offline (batch) plane**: generates synthetic data, prompts a teacher LLM, builds a distillation dataset, and produces QA reports. This is implemented today in `Data-Pipeline/` and orchestrated by an Airflow DAG.
- **Online (serving) plane**: a backend API and a deployed student model endpoint for interactive plan generation, coaching, and logging. This is planned for deployment on Google Cloud.

The offline plane supplies training data for the online plane (student model), and the online plane produces telemetry and interaction logs for evaluation and future iterations (with appropriate privacy controls).

## 1.1 Detailed Architecture Diagram (SVG)

![FitSenseAI Detailed Architecture Diagram](assets/fitsenseai_architecture.svg)

## 2. Current Repo Components (Implemented)

### 2.1 Data Pipeline (`Data-Pipeline/`)

Phases implemented and orchestrated:

1. bootstrap / reproducibility setup
2. synthetic profiles and health context generation
3. synthetic workout plans and execution logs generation
4. synthetic natural-language query generation (prompt types + slice tags)
5. teacher LLM calling + response capture (mock and openai-compatible providers)
6. distillation dataset builder (filtering + deterministic stratified splits)
7. QA: validation, statistics, anomaly detection, bias slicing

Airflow DAG:

- `Data-Pipeline/dags/fitsense_pipeline.py` (DAG ID: `fitsense_pipeline`)

### 2.2 Database Schema (`database/`)

- `database/tables.sql` defines the core relational schema for users, goals, workouts, health logs, and AI interactions.

## 3. Target Google Cloud Architecture (Planned)

### 3.1 High-Level Component Map

- **Client** (web/mobile): onboarding, plan view, workout logging, daily check-ins.
- **Backend API** (Cloud Run): auth, CRUD for plans/logs, inference routing, interaction logging.
- **Primary DB** (Cloud SQL for PostgreSQL): normalized user/workout/health tables.
- **Artifacts and Datasets** (Cloud Storage): pipeline artifacts, distillation dataset versions, reports.
- **Offline Orchestration** (self-managed Airflow on Compute Engine, or Cloud Scheduler + Workflows/Jobs): scheduled/offline pipeline runs.
- **Training and Registry** (Compute Engine / GCP Batch / GKE + GCS manifests): training jobs, model versioning, evaluation artifacts.
- **Serving** (Cloud Run for CPU-quantized models, or GKE/Compute Engine for GPU inference): student model inference.
- **Observability** (Cloud Logging/Monitoring): system metrics, traces, alerts.

## 4. Data Flow (End-to-End)

### 4.1 Offline Flow (Synthetic -> Distillation)

1. Generate synthetic structured tables aligned to `database/tables.sql`.
2. Generate synthetic natural-language prompts from structured user state.
3. Call teacher LLM and store responses + metadata.
4. Build distillation JSONL dataset and create deterministic train/val/test splits.
5. Run QA modules and emit reports (validation/stats/anomaly/bias).

### 4.2 Training Flow (Distillation -> Student Model)

1. Read a distillation dataset run from Cloud Storage.
2. Train (fine-tune) a student model on cost-controlled GCP compute (Compute Engine/GCP Batch/GKE).
3. Evaluate using curated prompt sets and slice-based checks.
4. Register model version via a lightweight registry approach (GCS artifact path + JSON manifest) with links to dataset run ids and evaluation results.

### 4.3 Online Flow (User -> Plan/Coach -> Logs)

1. User sends request (plan creation/modification/coaching) to Backend API.
2. Backend loads user context (goals, constraints, recent logs) from Cloud SQL.
3. Backend calls student inference endpoint.
4. Backend returns response to client and stores:
   - `ai_interactions` with model version and request context,
   - any created/updated plan entities,
   - workout logs and daily check-ins.

## 5. Key Interfaces (Contracts)

### 5.1 Offline Artifacts

- Synthetic datasets:
  - `Data-Pipeline/data/raw/synthetic_profiles/<run_id>/`
  - `Data-Pipeline/data/raw/synthetic_workouts/<run_id>/`
- Teacher inputs/outputs:
  - `Data-Pipeline/data/raw/synthetic_queries/<run_id>/`
  - `Data-Pipeline/data/raw/teacher_outputs/<run_id>/`
- Distillation dataset:
  - `Data-Pipeline/data/raw/distillation_dataset/<run_id>/` (JSONL splits)
- QA reports:
  - `Data-Pipeline/data/reports/phase6/<run_id>/` (JSON)

### 5.2 Online API (Planned)

- Endpoints (representative):
  - `POST /plans` (create from user context)
  - `POST /plans/{id}:modify` (edit plan)
  - `POST /coach` (Q&A / guidance)
  - `POST /workouts` and `POST /workouts/{id}/sets` (logging)
  - `POST /daily/sleep`, `POST /daily/calories`, `POST /daily/weight`

## 6. Cross-Cutting Concerns

- **Reproducibility**: deterministic seeds and stable split logic in the offline pipeline (`Data-Pipeline/params.yaml`).
- **Auditability**: store model name/version and request metadata for every AI response (`ai_interactions`).
- **Safety**: enforce conservative policy on pain/injury prompts; log safety flags; gate training on filtered teacher outputs.
- **Privacy**: per-user isolation and least-privilege IAM; no training on raw user chat without explicit opt-in (future).

## 7. Open Items (Decisions Needed)

- Final choice of:
  - student model family and serving mode (Cloud Run CPU-quantized vs GPU-backed serving on GKE/Compute Engine),
  - orchestration layer for cloud batch runs (self-managed Airflow vs Cloud Scheduler + Workflows/Jobs),
  - evaluation harness location and CI/CD integration.
