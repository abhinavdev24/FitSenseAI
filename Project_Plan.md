# FitSenseAI Project Plan (Updated)

## Project Status Snapshot

The repository now contains an implemented synthetic-data MLOps pipeline under `Data-Pipeline/` that covers:

- synthetic fitness/health data generation,
- synthetic query generation for teacher prompting,
- teacher LLM response capture,
- distillation dataset construction for student-model training,
- validation/statistics/anomaly detection/bias slicing,
- Airflow DAG orchestration (`fitsense_pipeline`),
- local pipeline documentation and diagrams.

This plan reflects the current repository state and the next execution phases.

---

## Team Roles (6-person split)

- **Data/MLOps (2):** data pipeline, Airflow, DVC, reproducibility, validation/monitoring, infrastructure deployment.
- **Machine Learning (2):** teacher prompting strategy, distillation dataset quality, student fine-tuning, evaluation and safety benchmarking.
- **Backend (1):** API services, business logic, persistence, inference integration.
- **Frontend/Mobile (1):** user onboarding, plan/logging UI, client integration with backend endpoints.

---

## Phase Breakdown and Current State

### Phase 1: Foundation & Data Pipeline Base (Completed in Repository)

This phase is implemented in `Data-Pipeline/` and includes the project’s synthetic-data-first pipeline foundation.

#### Completed work

1. **Pipeline scaffolding and reproducibility setup**
   - `Data-Pipeline/` folder structure (`dags/`, `scripts/`, `tests/`, `data/`, `logs/`)
   - `Data-Pipeline/params.yaml`
   - shared utilities in `Data-Pipeline/scripts/common/`
   - deterministic seed handling and bootstrap validation

2. **Synthetic source data generation (schema-aligned)**
   - `generate_synthetic_profiles.py`
   - `generate_synthetic_workouts.py`
   - outputs written to `Data-Pipeline/data/raw/synthetic_profiles/` and `Data-Pipeline/data/raw/synthetic_workouts/`

3. **Synthetic teacher-query generation**
   - `generate_synthetic_queries.py`
   - prompt types: plan creation, modification, safety adjustment, progress adaptation
   - metadata and slice tags included for downstream bias analysis

4. **Teacher LLM response pipeline**
   - `call_teacher_llm.py`
   - mock provider (deterministic) and `openai_compatible` provider support
   - retries, timeouts, request/response logging, post-validation, safety flags

5. **Distillation dataset build**
   - `build_distillation_dataset.py`
   - filtering and deterministic stratified train/val/test split generation
   - JSONL outputs under `Data-Pipeline/data/raw/distillation_dataset/`

6. **Validation and monitoring artifacts (data quality stage)**
   - `validate_data.py`
   - `compute_stats.py`
   - `detect_anomalies.py`
   - `bias_slicing.py`
   - reports under `Data-Pipeline/data/reports/phase6/`

7. **Airflow orchestration**
   - `Data-Pipeline/dags/fitsense_pipeline.py`
   - end-to-end DAG tested locally via `airflow dags test`

8. **Tests**
   - unit/integration tests across pipeline phases in `Data-Pipeline/tests/`

#### Dependencies closed in this phase

- Schema definition (`database/tables.sql`) -> synthetic generators
- Synthetic generators -> query generation
- Query generation -> teacher response capture
- Teacher responses -> distillation dataset build
- Distillation dataset -> validation/stats/anomaly/bias reports
- All scripts -> Airflow DAG orchestration

---

### Phase 2: Student Model Training Pipeline (In Progress / Next Implementation Block)

This phase uses the completed synthetic + teacher pipeline outputs to train the first student model (e.g., Gemma).

#### Tasks

1. **Training dataset contract finalization**
   - finalize instruction/context/response schema for SFT
   - lock tokenization and formatting convention (chat template vs plain instruction format)
   - version a training-ready dataset manifest from `distillation_dataset/<run_id>`

2. **Student fine-tuning pipeline**
   - training scripts and configs (LoRA/QLoRA or full fine-tune depending hardware)
   - reproducible train/eval runs with run metadata
   - checkpoint and artifact management

3. **Evaluation baseline**
   - prompt set for quality/safety checks
   - benchmark student vs teacher on representative slices (`goal_type`, `activity_level`, `condition_flag`)
   - report latency/quality tradeoffs

4. **Model registry/versioning**
   - version student checkpoints and eval results
   - tie model versions to distillation dataset run IDs

#### Dependencies

- Depends on Phase 1 outputs (`distillation_dataset`) and teacher-response artifacts.

---

### Phase 3: Backend API and Inference Integration (Planned)

#### Tasks

1. **Backend service foundation**
   - API service for onboarding, goals, workout plans, workout logging, and daily health logs
   - database integration based on `database/tables.sql`

2. **Inference integration**
   - student model inference endpoint integration
   - teacher fallback or offline evaluation endpoints (internal/admin use)
   - request/response logging with traceable model version metadata

3. **Plan generation and adaptation endpoints**
   - create plan
   - modify plan
   - safety-aware adjustment
   - progress adaptation

4. **Authentication and user isolation**
   - per-user data access boundaries
   - secret management for model provider keys and DB credentials

#### Dependencies

- Depends on Phase 2 student model v1 and database setup.

---

### Phase 4: Client MVP (Web/Mobile) (Planned)

#### Tasks

1. **Onboarding and goal capture UI**
2. **Workout plan viewing and execution logging UI**
3. **Daily logs UI (calories, sleep, weight)**
4. **Backend integration and error handling**
5. **Instrumentation hooks for usage analytics**

#### Dependencies

- Depends on Phase 3 API endpoints (mock API can be used for parallel UI work).

---

### Phase 5: Safety, Monitoring, and Hardening (Planned)

#### Tasks

1. **Model/application monitoring expansion**
   - production latency/error dashboards
   - model quality drift indicators
   - data drift against pipeline slice distributions

2. **Safety hardening**
   - prompt policy layers and refusal templates
   - unsafe output detection refinements
   - escalation handling for medical-risk prompts

3. **Security hardening**
   - secret rotation and environment separation
   - audit logging and access controls

4. **Operational robustness**
   - retries/backoff review across services
   - queueing/caching strategies
   - incident runbooks

#### Dependencies

- Depends on deployed backend/client/inference integration.

---

### Phase 6: Pilot, Iteration, and Finalization (Planned)

#### Tasks

1. **Pilot rollout**
   - limited user group deployment
   - usage monitoring and feedback collection

2. **Iteration cycle**
   - bug fixes
   - model quality improvements
   - UX refinements

3. **Final validation and reporting**
   - end-to-end evaluation against project criteria
   - documentation consolidation (`README.md`, `Data-Pipeline/README.md`, runbooks)
   - demonstration preparation

#### Dependencies

- Depends on completion of Phases 3-5.

---

## Current Deliverables in Repository (Implemented)

### Data/MLOps Pipeline

- `Data-Pipeline/dags/fitsense_pipeline.py`
- `Data-Pipeline/scripts/*.py` for Phases 1-6 data pipeline operations
- `Data-Pipeline/tests/` coverage for generators, teacher pipeline, distillation build, and monitoring modules
- `Data-Pipeline/docs_assets/` diagrams used in documentation

### Documentation

- `Data-Pipeline/README.md` (pipeline README/documentation)
- `README.md` (project overview with database + pipeline summary)
- `MLOPS-1-2_FitSenseAI_Execution_Guide.md` (implementation-focused execution guide)

### Data/Artifacts (local runs)

- synthetic raw datasets
- teacher outputs
- distillation dataset splits
- validation/stats/anomaly/bias reports
- Airflow local metadata (`.airflow/`) from DAG testing

---

## Near-Term Execution Order (Next Work)

1. Build student model training pipeline on top of `distillation_dataset` outputs.
2. Add evaluation harness and baseline metrics for student vs teacher outputs.
3. Integrate student model into backend inference API.
4. Continue client MVP integration and production-grade monitoring/security hardening.
