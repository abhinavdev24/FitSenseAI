# FitSenseAI Environment and DevOps (GCP, No Vertex AI)

Last updated: 2026-02-22

## 1. Environments

Minimum environments:

- `dev`: fast iteration, lower quotas
- `stage`: integration testing with pinned model versions
- `prod`: user-facing

## 2. Resource Layout (Suggested)

Per environment:

- Cloud Run service: `fitsense-backend`
- Cloud SQL instance: `fitsense-postgres`
- GCS bucket: `fitsenseai-mlops-{env}`
- Inference VM: `fitsense-inference-{env}` (private subnet)

## 3. Secrets and Config

- Store secrets in Secret Manager:
  - DB credentials
  - external teacher API key (offline)
  - backend auth secrets
- Do not store secrets in repo or in container images.

## 4. CI/CD (Suggested)

Cloud Build pipeline responsibilities:

- build backend container and push to Artifact Registry
- deploy backend to Cloud Run
- build training container and push to Artifact Registry
- optionally build pipeline job containers

## 5. Networking

- Use Serverless VPC Access connector for Cloud Run to reach:
  - Cloud SQL (private IP)
  - inference VM private IP
- Firewall:
  - allow inference port only from VPC connector subnet
  - allow SSH only from admin IPs

## 6. Operational Guardrails

- Backend:
  - rate limiting per user
  - max token caps and timeouts
  - bounded queues for inference
- Inference VM:
  - process supervision (systemd or container restart policy)
  - health checks
  - log forwarding to Cloud Logging

