# FitSenseAI

AI-powered fitness coaching application focused on personalized workouts, progress tracking, and health-aware guidance.

## What's In This Repository

This repository currently includes:

- project planning artifacts,
- database schema design for the FitSenseAI domain,
- an implemented synthetic-data MLOps pipeline in `Data-Pipeline/` (data generation → teacher LLM → distillation dataset → validation/monitoring),
- Airflow DAG orchestration for the pipeline,
- a model training and evaluation pipeline in `Model-Pipeline/` (fine-tuning → evaluation → bias analysis → GCS/Vertex AI registry).

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
  Project_Plan.md
  Project_Scoping.md
  setup_vm.sh
  Data-Pipeline/
    dags/
    scripts/
    tests/
    data/
    logs/
    params.yaml
    requirements.txt
    dvc.yaml
  Model-Pipeline/
    adapters/
      qwen3-4b-fitsense/        ← trained LoRA adapter (stored in GCS)
    data/
      formatted/                ← Qwen3 ChatML formatted training data
    reports/
      student_eval_*.json       ← per-sample evaluation results
      bias_report_*.json        ← demographic slice analysis
      registry_record_*.json    ← Vertex AI push metadata
    scripts/
      prepare_training_data.py  ← formats distillation dataset for training
      trainmodel.py             ← fine-tunes Qwen3-4B with LoRA via Unsloth
      evaluate_student.py       ← runs inference and computes ROUGE-L/BERTScore
      check_schema.py           ← validates JSON schema of model outputs
      bias_slicing.py           ← slices metrics by demographic attributes
      push_to_registry.py       ← packages and pushes adapter to GCS + Vertex AI
  database/
    database_design.dbml
    postgresql.sql
    mysql.sql
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
3. Use `database/postgresql.sql` (PostgreSQL) or `database/mysql.sql` (MySQL) as the base SQL export for database setup.
4. Prefer PostgreSQL for implementation (recommended for this project's relational complexity and future analytics needs).

## Planned Architecture (High Level)

- Backend API: user onboarding, goal capture, workout planning, logging, and AI endpoints.
- Mobile/Web client: plan viewing, workout execution logging, and daily check-ins (calories/sleep/weight).
- AI layer: plan generation/adaptation and conversational guidance.
- Data layer: relational DB for user/workout/health data and model interaction logs.

See `Project_Plan.md` for phase-wise execution details.

## Architecture Diagram (Detailed)

For a more complete write-up, see `docs/Architecture.md`.

![FitSenseAI Detailed Architecture Diagram](docs/assets/fitsenseai_architecture.svg)

## Data Pipeline (Overview)

FitSenseAI includes an end-to-end synthetic-data MLOps pipeline under `Data-Pipeline/` that covers:

- synthetic profile/workout/health data generation,
- synthetic query generation for a teacher LLM (Qwen 32B via Groq),
- teacher response capture and storage,
- distillation dataset creation (train/val/test JSONL),
- validation, statistics, and anomaly detection,
- Airflow DAG orchestration for the full workflow.

Primary docs:

- `Data-Pipeline/README.md` for the full pipeline usage and Airflow commands.

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
```

## Model Pipeline (Overview)

FitSenseAI includes a model training and evaluation pipeline under `Model-Pipeline/` that covers:

- formatting the distillation dataset into Qwen3 ChatML format with `/no_think`,
- fine-tuning Qwen3-4B (student model) using LoRA adapters via Unsloth on a GPU instance,
- evaluating the student model with ROUGE-L and BERTScore metrics,
- validating JSON schema correctness of generated workout plans,
- analyzing performance across demographic slices (age, sex, goal type, activity level, medical conditions) for bias detection,
- packaging and pushing the trained adapter to GCS and registering it in Vertex AI Metadata Store.

### Model Architecture

| Component | Details |
|---|---|
| Teacher model | Qwen 32B via Groq API |
| Student model | Qwen3-4B fine-tuned with LoRA |
| Adapter type | LoRA (rank 16, alpha 32) |
| Training steps | 60 |
| Quantization | 4-bit NF4 (Unsloth) |
| Training data | Distillation dataset from Data-Pipeline |

### Evaluation Results (Run 20260403Z)

| Metric | Value | Threshold | Status |
|---|---|---|---|
| JSON validity | 45% | ≥ 50% | ⚠️ |
| Schema validity (overall) | 55% | ≥ 50% | ✅ |
| Schema validity (plan_creation) | 90.91% | — | ✅ |
| Schema validity (plan_updation) | 11.11% | — | ⚠️ |
| ROUGE-L | 0.1734 | ≥ 0.10 | ✅ |

### Model Registry

Trained adapter is stored in GCS and registered in Vertex AI:

- GCS: `gs://fitsense-adapter-store/models/fitsense-qwen3-4b/`
- Vertex AI Metadata Store: `projects/584243823383/locations/us-central1/metadataStores/default/artifacts/`

### Model Pipeline Quick Start

```bash
# 1. Format training data
python Model-Pipeline/scripts/prepare_training_data.py

# 2. Fine-tune student model (requires GPU — use Colab Pro or GCP GPU VM)
python Model-Pipeline/scripts/trainmodel.py

# 3. Evaluate student model
python Model-Pipeline/scripts/evaluate_student.py

# 4. Validate JSON schema
python Model-Pipeline/scripts/check_schema.py

# 5. Run bias analysis
python Model-Pipeline/scripts/bias_slicing.py

# 6. Push adapter to GCS + Vertex AI
python Model-Pipeline/scripts/push_to_registry.py
```

## Roadmap Snapshot

- Phase 1: Foundation and data schemas ✅
- Phase 2: Model development (teacher/student workflow) ✅
- Phase 3: Backend + app MVP
- Phase 4: Adaptation engine and instrumentation
- Phase 5: Safety, monitoring, hardening
- Phase 6: Pilot, iteration, final validation