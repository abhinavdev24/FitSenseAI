# FitSenseAI Development Phases

Last updated: 2026-02-22

This is the engineering delivery breakdown aligned to current repo state and the MVP target.

## Phase 1: Offline Pipeline (Completed in Repo)

Delivered in `Data-Pipeline/`:

- synthetic data generation
- synthetic prompt generation
- teacher calling + response capture
- distillation dataset build with deterministic splits
- QA reports: validation/stats/anomaly/bias slicing
- Airflow DAG orchestration
- tests

## Phase 2: Training and Evaluation (Next)

Deliverables:

- containerized training entrypoint for student fine-tuning
- evaluation harness and thresholds (safety + quality)
- model artifact and eval report stored in GCS
- model version manifest (dataset run_id, git SHA, config, eval URI)

## Phase 3: Private Serving (Next)

Deliverables:

- Compute Engine inference VM with single GPU
- vLLM server configured for 4k context and streaming
- private networking from Cloud Run to the VM via VPC connector
- rate limiting and backpressure/queueing in backend

## Phase 4: Backend API MVP (Next)

Deliverables:

- plan endpoints
- workout and daily logging endpoints
- coaching endpoint with streaming
- interaction logging with model metadata

## Phase 5: Hardening and Monitoring (MVP+)

Deliverables:

- dashboards for latency and error rates
- cost monitoring and quotas
- safety regression tests integrated into CI
- incident runbook basics

