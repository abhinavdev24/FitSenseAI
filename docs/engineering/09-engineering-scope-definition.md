# FitSenseAI Engineering Scope Definition (MVP)

Last updated: 2026-02-22

## 1. In Scope

- Keep and extend the existing offline pipeline under `Data-Pipeline/` as the training data source.
- External teacher API is used offline for synthetic prompt responses.
- Train a student model on a single cost-effective GPU (Compute Engine or GCP Batch).
- Serve the student model privately using vLLM on a single GPU inference VM.
- Backend API on Cloud Run:
  - plan generation and modification,
  - workout and daily logging,
  - adaptation guidance,
  - coaching chat with streaming.
- Data persistence in Cloud SQL aligned to `database/tables.sql`.
- Minimal observability (logs, metrics, basic alerts).

## 2. Out of Scope (MVP)

- Vertex AI services (training, registry, endpoints).
- Public model access.
- Rich analytics and BI dashboards.
- Full prompt-injection hardening beyond basic policies and private-only model access.
- Real-user-data training loops (synthetic-first remains default).

## 3. Constraints

- Single GPU serving requires:
  - strict token limits,
  - queueing/backpressure,
  - low true GPU concurrency (2-3).
- Cloud costs must be predictable and controllable.

## 4. Definition of Done (Engineering)

- Training job produces a versioned model artifact + eval report + manifest in GCS.
- Inference VM can serve streaming responses behind private networking.
- Backend integrates streaming responses end-to-end and logs model/version metadata.
- End-to-end minimal load test passes at peak concurrency assumptions via queueing/backpressure.

