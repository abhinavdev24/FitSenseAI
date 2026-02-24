# FitSenseAI Project Scoping

## 1. Introduction

FitSenseAI is an AI-powered fitness coaching application that delivers personalized, adaptive, and explainable workout guidance. The system is designed around a teacher-student LLM workflow, with a larger teacher model used to generate high-quality synthetic coaching outputs and a smaller student model targeted for efficient deployment.

Core capabilities in scope:

- goal-based workout planning,
- weekly adaptation based on user performance and recovery signals,
- natural-language coaching interactions,
- exercise execution guidance,
- safety-aware responses for user constraints,
- training-data generation for a deployable student model.

## 2. Problem Statement

Users commonly struggle to convert goals (fat loss, muscle gain, strength, general fitness) into structured weekly programs that adapt to real-world execution.

Key gaps in current user workflows:

- inconsistent plan quality across public sources,
- poor personalization for injuries/constraints/equipment availability,
- weak feedback loops from logged performance to next-week programming,
- high risk of contradictory guidance,
- limited explainability in generic chatbot-style advice.

## 3. Proposed Solution

FitSenseAI combines deterministic training rules with LLM-based personalization.

Solution components:

- **Hybrid planning engine**:
  - deterministic rules for progression, recovery, and safety boundaries,
  - LLM for personalization, explanation, and natural-language plan edits.
- **Teacher-student model strategy**:
  - teacher model generates synthetic coaching outputs,
  - student model is fine-tuned on curated prompt-response pairs for lower-latency deployment.
- **User-grounded coaching loop**:
  - plans, logs, sleep, calorie intake, and weight tracking support iterative adaptation.
- **Safety-first behavior**:
  - explicit handling of injury/condition-related prompts with conservative guidance and escalation language.

## 4. Scope

### 4.1 In Scope

- Personalized workout plan generation and modification
- Workout execution logging (sets, reps, weight, RIR, notes)
- Daily health-adjacent tracking (calories, sleep, weight)
- Teacher-student synthetic training data pipeline
- Distillation dataset creation for student-model training
- Data validation, anomaly detection, and bias slicing on pipeline outputs
- Airflow orchestration for the data pipeline workflow

### 4.2 Out of Scope (Current Project Stage)

- Medical diagnosis or treatment recommendations
- Detailed macro tracking (protein/carbs/fats)
- Hydration tracking
- Production-grade video generation service (prompt/caching strategy may be scoped, full generation system can be phased later)
- Clinical-grade decision support

## 5. Data Strategy (Synthetic-First)

### 5.1 Data Approach

The project uses a **synthetic-data-first pipeline** for model-training and pipeline validation workflows. The system does not depend on external datasets for the core MLOps pipeline design.

Planned data categories:

- synthetic user profiles and health context,
- synthetic workout plans and workout execution logs,
- synthetic daily calorie/sleep/weight logs,
- synthetic natural-language user prompts,
- teacher LLM responses,
- distilled instruction-tuning dataset for the student model.

### 5.2 Why Synthetic-First

- enables deterministic reproducibility in the early MLOps pipeline,
- avoids licensing/privacy complexity during initial system validation,
- supports controlled slice coverage for fairness/bias analysis,
- allows targeted generation of safety-critical prompt scenarios.

### 5.3 Data Rights and Privacy Position

Even with synthetic-first training data, the product is designed for eventual user-generated data ingestion. The data policy baseline is:

- explicit consent for stored user logs and feedback,
- per-user isolation,
- encryption at rest/in transit,
- data minimization,
- deletion/export workflow support,
- no training on raw user chat logs without explicit opt-in and anonymization.

## 6. Data Planning and Splits

### 6.1 Synthetic Data Generation Planning

Planned structured outputs include:

- user/profile tables,
- goals/conditions and medical context,
- workout plans and executed workout logs,
- calorie/sleep/weight tracking logs.

### 6.2 Synthetic Prompt Generation Planning

Prompt types planned for teacher generation:

- plan creation,
- plan modification,
- safety adjustment,
- progress adaptation.

Each prompt includes metadata for downstream analysis:

- user/scenario identifiers,
- prompt type,
- slice tags (age band, sex, goal type, activity level, condition flag),
- expected safety constraints,
- compact context summary.

### 6.3 Distillation Dataset Splits

The student-training dataset is split into train/validation/test partitions with deterministic assignment and stratification by key attributes (e.g., prompt type and goal type) to preserve coverage balance.

## 7. Repository and Documentation Scope

Planned repository scope includes:

- project planning and scoping documents,
- database schema artifacts,
- data pipeline code and orchestration (`Data-Pipeline/`),
- tests and validation modules,
- diagrams and operational documentation.

Planned documentation set includes:

- root project README,
- pipeline-specific README,
- data pipeline write-up,
- project plan and scoping artifacts.

## 8. Current Approach Flow and Bottlenecks (Target User Journey)

### 8.1 Typical Current User Flow (Without FitSenseAI)

- user searches mixed sources for workouts,
- selects a generic plan,
- logs inconsistently (or does not log at all),
- has no structured adaptation loop,
- experiences plateau/confusion/risk from poor progression decisions.

### 8.2 Main Bottlenecks

- information overload and contradictions,
- no personalization loop from performance -> next plan,
- no unified tracking + coaching interface,
- unsafe guidance risk in generic chatbot responses,
- high cost of repeatedly generating media assets without caching.

### 8.3 FitSenseAI Improvements

- single loop: plan -> log -> adapt -> explain,
- personalized responses grounded in user context,
- consistent safety-aware guidance templates and constraints,
- lower-cost deployable model through distillation.

## 9. Metrics, Objectives, and Business Goals

### 9.1 Product and Model Metrics

**Personalization / Plan Quality**

- Plan acceptance rate (minimal user edits)
- Weekly adherence (completed sessions / planned sessions)
- Progress proxies (volume trend, estimated strength trend, reps at fixed load)

**Safety**

- Unsafe response rate on safety test prompts
- Injury/pain escalation handling correctness

**Chat / Coaching Quality**

- Grounded answer quality score (for future grounded modes)
- Hallucination rate on curated prompts

**System / Cost**

- p95 latency (plan generation / coaching interactions)
- Cost per active user (inference + storage + optional media generation)

### 9.2 Objectives

- Generate a usable workout plan within target latency
- Improve adherence vs. static-plan baseline
- Maintain unsafe response rate below defined threshold
- Keep deployment cost predictable via student-model serving

### 9.3 Business Goals

- Improve retention through weekly adaptation loops
- Differentiate via explainable, personalized coaching
- Achieve efficient deployment with a small student model

## 10. Failure Analysis and Risk Register

### 10.1 Project Execution Risks

- weak evaluation design leading to misleading quality conclusions,
- poor coverage in synthetic prompt generation,
- safety regressions in teacher outputs contaminating distillation data,
- pipeline reproducibility drift across environments.

### 10.2 Post-Deployment Risks

- model drift as user behavior changes,
- prompt injection or attempts to override safety constraints,
- privacy incidents from logging/storage misconfiguration,
- cost spikes from unbounded usage patterns.

### 10.3 Mitigation Strategy

- deterministic pipeline configuration and versioned artifacts,
- teacher-output filtering + post-validation + safety flags,
- curated evaluation sets and slice-based analysis,
- monitoring for schema shifts, anomalies, and distribution changes,
- secret management and per-user access controls.

## 11. Deployment Infrastructure (Planned)

### 11.1 Training / Batch Pipeline

- Data pipeline orchestration with Airflow
- Synthetic generation and teacher-calling jobs
- Distillation dataset build and validation jobs
- Student fine-tuning jobs (later phase)

### 11.2 Online Inference (Planned)

- Backend API service for onboarding, planning, logging, adaptation
- Student-model inference endpoint
- Data store for user/workout/health logs
- Optional retrieval layer and media caching services in later phases

### 11.3 Client Applications (Planned)

- Web/mobile client for goals, plans, logging, and coaching interactions

## 12. Monitoring Plan (Planned)

### 12.1 Data Monitoring

- schema changes,
- missing fields,
- duplicates,
- response-length anomalies,
- split imbalance in distillation datasets.

### 12.2 Bias / Slice Monitoring

- performance/quality proxies across:
  - age band,
  - sex,
  - goal type,
  - activity level,
  - condition flag.

### 12.3 System Monitoring

- task failures and retries,
- pipeline latency by stage,
- inference latency and error rates,
- cost monitoring (serving and storage).

## 13. Success and Acceptance Criteria

A release candidate is considered acceptable when:

### Functional

- users can generate and modify plans,
- users can log workouts and daily metrics,
- adaptation loop produces next-step guidance,
- student-model training data pipeline completes reproducibly end-to-end.

### Quality / Safety

- plan quality metrics meet defined thresholds,
- unsafe response rate remains below threshold on safety prompts,
- bias slicing checks do not show unaddressed severe disparities under defined thresholds.

### Operational

- Airflow pipeline runs reliably with documented setup,
- artifacts and reports are versionable/reproducible,
- no critical privacy/security gaps in scoped deployment design.

## 14. Timeline Plan (Project-Level)

### Phase 1 (Completed / Foundation)

- repository setup,
- schema definition,
- synthetic data pipeline and Airflow DAG,
- validation/monitoring report generation.

### Phase 2 (Next)

- student fine-tuning pipeline,
- evaluation baseline and model versioning.

### Phase 3

- backend API and inference integration.

### Phase 4

- client MVP (web/mobile) and integration.

### Phase 5

- safety, monitoring, and security hardening.

### Phase 6

- pilot, iteration, and final validation.

## 15. Additional Information

### Safety Position

FitSenseAI is not a medical device and does not diagnose or treat injuries or medical conditions.

For pain/injury/medical-condition prompts, the system provides conservative general guidance and escalation language encouraging professional consultation where appropriate.

### Model Strategy Summary

- **Teacher LLM:** used to generate diverse, high-quality synthetic coaching outputs and training targets.
- **Student LLM:** optimized for lower-latency, predictable-cost deployment after distillation.
