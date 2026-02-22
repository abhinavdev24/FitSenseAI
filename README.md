# FitSenseAI

AI-powered fitness coaching application focused on personalized workouts, progress tracking, and health-aware guidance.

## What’s In This Repository

This repository currently includes:

- project planning artifacts,
- database schema design for the FitSenseAI domain,
- an implemented synthetic-data MLOps pipeline in `Data-Pipeline/` (data generation -> teacher LLM -> distillation dataset -> validation/monitoring),
- Airflow DAG orchestration for the pipeline.

## Project Scope

FitSenseAI is designed to:

- Create and adapt workout plans based on user goals, conditions, and workout history.
- Log workout execution (exercises, sets, reps, weight, RIR, notes).
- Store relevant user health context (medical profile, medications, allergies, injuries).
- Recommend daily maintenance calories and allow optional daily calorie intake logging.
- Recommend target sleep duration and allow optional daily sleep-duration logging.
- Allow users to log body weight any time for progress tracking.

Out of scope for current data model:

- Detailed nutrition macro tracking (protein/carbs/fats).
- Hydration tracking.

## Repository Structure

```txt
FitSenseAI/
  README.md
  FitSense_AI_Project_Plan.md
  FitSense_AI_Project_Scoping_Complete-1.pdf
  MLOPS-1-2_FitSenseAI_Execution_Guide.md
  Data-Pipeline/
    dags/
    scripts/
    tests/
    data/
    logs/
    params.yaml
    requirements.txt
    dvc.yaml
  database/
    database_design.dbml
    tables.sql
    UML_diagram.png
```

## Database Design

Primary schema file:

- `database/database_design.dbml`

Live ER diagram (no upload needed):

- https://dbdiagram.io/d/FitSenseAI-69850002bd82f5fce2cfe02c

Core model areas:

- User and goals: `users`, `goals`, `user_goals`
- Health context: `conditions`, `user_conditions`, `user_profiles`, `user_medical_profiles`, `user_medications`, `user_allergies`
- Workouts: `workout_plans`, `plan_exercises`, `plan_sets`, `workouts`, `workout_exercises`, `workout_sets`
- Guidance + tracking:
  - Calories: `calorie_targets`, `calorie_intake_logs`
  - Sleep: `sleep_targets`, `sleep_duration_logs`
  - Weight: `weight_logs`
- AI interactions: `ai_interactions`

## Quick Start (Schema)

1. Open the live diagram: [UML Diagram](https://dbdiagram.io/d/FitSenseAI-69850002bd82f5fce2cfe02c)
2. No DBML upload is required for viewers; they can access the schema directly from the link.
3. Use `database/tables.sql` as the base SQL export for database setup.
4. Prefer PostgreSQL for implementation (recommended for this project’s relational complexity and future analytics needs).

## Planned Architecture (High Level)

- Backend API: user onboarding, goal capture, workout planning, logging, and AI endpoints.
- Mobile/Web client: plan viewing, workout execution logging, and daily check-ins (calories/sleep/weight).
- AI layer: plan generation/adaptation and conversational guidance.
- Data layer: relational DB for user/workout/health data and model interaction logs.

See `FitSense_AI_Project_Plan.md` for phase-wise execution details.

## Architecture Diagram (Detailed)

For a more complete write-up, see `docs/Architecture.md`.

![FitSenseAI Detailed Architecture Diagram](docs/assets/fitsenseai_architecture.svg)

## Data Pipeline (Overview)

FitSenseAI includes an end-to-end synthetic-data MLOps pipeline under `Data-Pipeline/` that covers:

- synthetic profile/workout/health data generation,
- synthetic query generation for a teacher LLM,
- teacher response capture and storage,
- distillation dataset creation (train/val/test JSONL),
- validation, statistics, anomaly detection, and bias slicing,
- Airflow DAG orchestration for the full workflow.

Primary docs:

- `Data-Pipeline/README.md` for the full pipeline usage and Airflow commands

### Pipeline Component Diagram

![FitSenseAI Data Pipeline Components](Data-Pipeline/docs_assets/pipeline_components.svg)

### Airflow DAG Diagram

![FitSenseAI Airflow DAG](Data-Pipeline/docs_assets/fitsense_pipeline_dag.svg)

### Pipeline Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Data-Pipeline/requirements.txt

# bootstrap
python Data-Pipeline/scripts/bootstrap_phase1.py

# run pipeline stages (script-by-script)
python Data-Pipeline/scripts/generate_synthetic_profiles.py
python Data-Pipeline/scripts/generate_synthetic_workouts.py
python Data-Pipeline/scripts/generate_synthetic_queries.py
python Data-Pipeline/scripts/call_teacher_llm.py
python Data-Pipeline/scripts/build_distillation_dataset.py
python Data-Pipeline/scripts/validate_data.py
python Data-Pipeline/scripts/compute_stats.py
python Data-Pipeline/scripts/detect_anomalies.py
python Data-Pipeline/scripts/bias_slicing.py
```

Bootstrap outputs:

- `Data-Pipeline/data/reports/phase1_bootstrap.json`
- `Data-Pipeline/logs/pipeline.log`

## Roadmap Snapshot

- Phase 1: Foundation and data schemas
- Phase 2: Model development (teacher/student workflow)
- Phase 3: Backend + app MVP
- Phase 4: Adaptation engine and instrumentation
- Phase 5: Safety, monitoring, hardening
- Phase 6: Pilot, iteration, final validation
