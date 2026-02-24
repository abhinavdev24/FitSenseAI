# FitSenseAI Product Requirements (Engineering)

Last updated: 2026-02-22

This document is the engineering-facing requirements spec for the FitSenseAI MVP. It is derived from:

- `docs/PRD.md`
- `docs/Architecture.md`
- `docs/System_Design.md`
- `docs/MVP_Tech_Doc.md`

## 1. Product Summary

FitSenseAI is a fitness coaching app that:

- generates personalized weekly workout plans,
- supports structured workout and daily signal logging,
- adapts guidance week-over-week,
- behaves conservatively in injury/medical-risk scenarios,
- uses a teacher-student workflow to reduce inference cost via a fine-tuned student model.

## 2. MVP Goals

- Initial plan generation from goal + constraints.
- Plan modification by user request and safety constraints.
- Workout logging (sets/reps/weight/RIR/notes).
- Optional daily logs (sleep, calories, weight).
- Adaptation output based on adherence and basic trends.
- Student model training pipeline that consumes the distillation dataset produced by `Data-Pipeline/`.
- Private-only LLM serving behind the backend API.

## 3. Non-Goals (MVP)

- Medical diagnosis or treatment.
- Detailed nutrition macro tracking and hydration tracking.
- Video generation or production media workflows.
- Public model endpoint access (model is backend-only).

## 4. Personas (MVP)

- Beginner: wants simple plan and execution clarity.
- Intermediate: wants progression logic and adaptability.
- Constraint-heavy: needs safe modifications and clear boundaries.

## 5. Functional Requirements

### 5.1 Onboarding and Profile

- Capture: goal type, schedule (days/week), equipment, experience level, constraints/injury flags.
- Store user profile in relational DB aligned to `database/tables.sql`.

### 5.2 Plan Generation

- Generate a weekly plan as structured output (sessions, exercises, target sets/reps, rest, target RIR).
- Include a short explanation of progression and safety notes.
- Create `workout_plans`, `plan_exercises`, `plan_sets` rows in DB.

### 5.3 Plan Modification

- Accept user instructions (swap exercise, adjust volume/intensity, change schedule).
- Maintain safety constraints (avoid contraindicated movements; cap effort; provide conservative alternatives).
- Persist plan revisions (MVP can overwrite active plan; later versions can snapshot).

### 5.4 Workout Logging

- Create workout sessions and log completion at set granularity:
  - `workouts`, `workout_exercises`, `workout_sets`.
- Support attaching a workout to an existing plan (`plan_id` linkage).

### 5.5 Daily Logs

- Optional logs:
  - calories (total),
  - sleep duration,
  - weight.
- Store aligned to `calorie_intake_logs`, `sleep_duration_logs`, `weight_logs`.

### 5.6 Adaptation Loop

Given the last N workouts and optional daily logs:

- compute adherence (completed/planned),
- detect basic progression/plateau indicators (volume/load/RIR trends),
- output a next-week adjustment recommendation, and optionally propose plan edits.

### 5.7 Coaching Chat (Scoped)

- Provide natural-language answers limited to fitness coaching.
- For pain/injury/medical prompts: conservative guidance + escalation language.
- Log interactions to `ai_interactions` with model name/version metadata.

## 6. Non-Functional Requirements

### 6.1 Performance

- Streaming responses from the LLM.
- Enforce per-request limits:
  - context window target: 4096 tokens,
  - output cap target: 1024 tokens (backend-enforced).

### 6.2 Reliability

- Graceful overload:
  - queue requests (preferred) or return `429` when saturated.
- Clear timeouts and retries for external teacher API calls (offline pipeline).

### 6.3 Cost Controls

- Student model serves most requests.
- Teacher model used primarily offline for data generation (external API).
- Single GPU inference is capacity-limited by design; keep true GPU concurrency low and queue.

### 6.4 Security and Privacy

- Private model endpoint (backend-only).
- Secrets stored in Secret Manager.
- Least-privilege IAM for Cloud SQL and GCS.
- No training on raw user chat logs without explicit opt-in (future).

## 7. System Requirements (GCP, No Vertex AI)

### 7.1 Online Plane

- Backend: Cloud Run.
- DB: Cloud SQL (PostgreSQL).
- LLM serving: Compute Engine VM (1 GPU) running vLLM, private subnet, reachable only from backend via VPC connector.

### 7.2 Offline Plane

- Data pipeline: current `Data-Pipeline/` Airflow DAG can be run locally; for cloud, run as batch jobs + orchestration (self-managed Airflow or Cloud Scheduler + Workflows + Jobs).
- Training: Compute Engine (GPU) or GCP Batch (GPU) running containerized training jobs.
- Artifacts: GCS buckets for datasets, reports, and model versions.

## 8. Acceptance Criteria (MVP)

- Offline pipeline produces a distillation dataset with deterministic splits and QA reports (already implemented in `Data-Pipeline/`).
- Student training job can consume a specific distillation `run_id` from GCS and emit model artifacts + eval report.
- Backend can:
  - generate/modify plans using the student model,
  - stream responses to the client,
  - log workouts and daily signals,
  - persist `ai_interactions` with model metadata.

