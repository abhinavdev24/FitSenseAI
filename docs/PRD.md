# FitSenseAI PRD (Product Requirements Document)

Version: 0.1 (MVP)
Last updated: 2026-02-22

## 1. Overview

FitSenseAI is an AI-powered fitness coaching application that generates personalized workout plans, supports structured logging (workouts + daily signals), and adapts programming week-over-week with safety-aware guidance.

This project is built around a teacher-student LLM workflow:

- a larger teacher model generates high-quality coaching outputs for synthetic prompts,
- a smaller student model is fine-tuned (distilled) for low-latency, predictable-cost deployment.

The repository currently contains an implemented synthetic-data MLOps pipeline under `Data-Pipeline/` that generates synthetic user/workout/health data, produces teacher prompts, captures teacher outputs, builds a distillation dataset, and runs QA (validation/stats/anomaly detection/bias slicing) with Airflow orchestration.

## 2. Problem Statement

People struggle to convert fitness goals into structured, safe, and adaptive weekly programs, especially when real-world constraints exist (injuries, equipment, schedule variability). Generic plans and chatbots typically fail on:

- personalization depth (constraints, preferences, history),
- consistent adaptation from logged performance,
- explainability and actionable next steps,
- safety boundaries (pain/injury/medical risk scenarios).

## 3. Target Users (Personas)

- Beginner lifter: wants a simple, structured plan; needs confidence and clarity.
- Intermediate lifter: wants progression logic and tweaks based on performance.
- Constraint-heavy user: has injuries/conditions/equipment limitations; needs conservative adjustments and clear safety language.

## 4. Goals and Non-Goals

### 4.1 Goals (MVP)

- Generate a personalized weekly workout plan from goal + constraints.
- Allow users to log workouts (sets/reps/weight/RIR/notes).
- Allow optional daily logs (sleep, calories, weight) to support adaptation signals.
- Adapt the plan (or generate next-week guidance) using logged history.
- Provide safety-aware responses when constraints or injury-related prompts occur.
- Produce a training-ready distillation dataset to train a deployable student model.

### 4.2 Non-Goals (MVP)

- Medical diagnosis/treatment recommendations.
- Detailed nutrition macro tracking (protein/carbs/fats) and hydration tracking.
- Production-grade media generation pipeline (may be phased later).
- Fully grounded retrieval over external exercise literature (may be phased later).

## 5. MVP Scope (What “Done” Means)

### 5.1 User-Facing Capabilities

- Onboarding: capture goal, experience level, schedule, equipment, constraints/injuries (high-level).
- Plan generation: create an initial weekly plan with explanation and safety notes.
- Logging: record workouts and daily signals (sleep/calories/weight).
- Adaptation: produce next-week plan adjustments based on adherence and simple performance proxies.
- Coaching: natural-language Q&A limited to fitness coaching scope with conservative safety behavior.

### 5.2 System Capabilities (MVP)

- Offline pipeline: synthetic data -> teacher prompts -> teacher outputs -> distillation dataset -> QA reports.
- Model training: student fine-tuning on Google Cloud using cost-controlled compute (Compute Engine/GCP Batch/GKE) and the distillation dataset.
- Online inference: student model serving endpoint and a backend API that logs interactions with model/version metadata.

## 6. Functional Requirements

### 6.1 Plan Generation and Editing

- Create plan from:
  - goal type (fat loss, muscle gain, strength, general fitness),
  - equipment availability,
  - time constraints (days/week),
  - injury/condition flags.
- Modify plan based on:
  - user request (swap exercise, adjust intensity, change schedule),
  - safety constraints (avoid aggravating movements).

### 6.2 Workout Logging

- Record completed session details aligned to the relational schema in `database/tables.sql`:
  - workout session, exercises, sets, reps, weight, RIR, notes.

### 6.3 Daily Signal Logging

- Optional logs:
  - sleep duration,
  - calorie intake (total only),
  - body weight.

### 6.4 Adaptation Loop

- Generate next-step guidance using:
  - adherence (completed/planned),
  - volume/load trend,
  - subjective difficulty proxy (RIR trend),
  - simple recovery proxies (sleep trend, optional).

### 6.5 Safety Behavior

- Conservative guidance for injury/pain prompts with escalation language.
- Refuse or redirect non-fitness/medical advice requests outside scope.
- Log safety flags in model outputs for auditing.

## 7. Non-Functional Requirements

- Latency: student inference should target interactive UX (e.g., sub-second to a few seconds depending on deployment mode).
- Cost: predictable per-request inference costs (primary motivation for student model).
- Reliability: clear error handling and retries for batch pipeline tasks; online endpoints should be resilient.
- Auditability: tie each response to a model version and request metadata for debugging and evaluation.
- Privacy and security: encryption in transit/at rest; secrets managed centrally; least-privilege IAM.

## 8. Success Metrics

### 8.1 Product Metrics

- Plan acceptance rate (users accept with minimal edits).
- Weekly adherence (completed sessions / planned sessions).
- Retention proxy: week-2 and week-4 active rate (post-MVP instrumented).

### 8.2 Safety Metrics

- Unsafe response rate on curated safety prompts below a defined threshold.
- Correct escalation handling rate for injury/medical-risk prompts.

### 8.3 System Metrics

- p95 latency and error rate of inference endpoints.
- Pipeline run success rate and runtime by stage.
- Cost per active user and cost per training run (GCP).

## 9. Data and Model Requirements

### 9.1 Data Strategy

- Synthetic-first pipeline is the default for MLOps validation and early model training.
- Synthetic prompt types: plan creation, plan modification, safety adjustment, progress adaptation.
- Distillation dataset outputs are JSONL with deterministic, stratified train/val/test splits.

### 9.2 Model Strategy

- Teacher model:
  - generates training targets for distillation (external provider or self-hosted open-weights model).
  - outputs are filtered/validated before inclusion in training.
- Student model:
  - fine-tuned on distillation dataset for efficient deployment.
  - evaluated against a safety/quality harness before serving.

## 10. Dependencies and Assumptions

- Deployment and training use Google Cloud:
  - Cloud Storage for artifacts and datasets,
  - Cloud Run (or similar) for backend API.
  - cost-controlled compute (Compute Engine/GCP Batch/GKE) for training (and optionally GPU inference).
- The current repo has a working local Airflow DAG for the offline pipeline; cloud orchestration will be a later implementation step (e.g., Cloud Composer).
- Schema and domain entities follow `database/tables.sql`.

## 11. Risks

- Synthetic coverage gaps leading to weak generalization.
- Teacher output contamination (unsafe/low-quality) leaking into distillation dataset.
- Evaluation design not representative of real usage.
- Cost spikes if online inference lacks quotas/caching/guardrails.

## 12. Milestones (High-Level)

1. MVP offline pipeline complete (already implemented under `Data-Pipeline/`).
2. Student training pipeline on cost-controlled GCP compute with evaluation baseline.
3. Backend API + student inference integration on GCP.
4. Client MVP for onboarding, plan display, logging, and adaptation.
