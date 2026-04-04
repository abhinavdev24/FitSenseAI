# Student LLM integration patch

This repository was patched so the backend can automatically use the trained student model once adapter weights are present.

## Files read before patching

- `README.md`
- `UPDATED_README.md`
- `docs/Model_Pipeline_Plan.md`
- `Model-Pipeline/scripts/prepare_training_data.py`
- `Model-Pipeline/scripts/trainmodel.py`
- `Model-Pipeline/scripts/evaluate_student.py`
- `Model-Pipeline/scripts/push_to_registry.py`
- `backend/app/services.py`
- `backend/app/main.py`
- `backend/README.md`

## What those files showed

- The student model is a LoRA-fine-tuned Qwen3-8B adapter trained from teacher-generated distillation data.
- Training format uses Qwen ChatML plus `/no_think` and an empty `<think></think>` block.
- Evaluation loads a base model plus adapter path from `Model-Pipeline/adapters/qwen3-8b-fitsense`.
- Registry records store the base model metadata and run metadata.
- The backend originally used rule-based logic for plan creation, plan modification, and coaching.

## Where the backend was patched

### 1. `backend/app/llm_runtime.py`
Added a lazy runtime that:
- discovers a trained adapter automatically under `Model-Pipeline/adapters/`
- optionally reads registry metadata under `Model-Pipeline/reports/registry_record_*.json`
- loads the base model plus LoRA adapter only when needed
- falls back cleanly when weights or dependencies are missing

### 2. `backend/app/services.py`
Added:
- `try_student_plan_generation(...)`
- `try_student_coach_reply(...)`
- `_plan_request_prompt(...)`
- `_create_plan_from_llm_json(...)`

Patched these flows to try the student model first:
- `generate_plan(...)`
- `modify_plan(...)`
- `build_coach_reply(...)`

### 3. `backend/app/main.py`
Added:
- `GET /model/runtime`

Also changed AI interaction logging so coach calls record the student model name when the student runtime is active.

### 4. `backend/requirements-llm.txt`
Added optional inference dependencies so model serving can be enabled later without changing application code.

## Important limitation

The uploaded codebase documents the student-model training pipeline, but it does **not** include actual adapter weight files. Because of that, this patch cannot guarantee real student-model inference immediately on a fresh clone. Instead, it guarantees that **when adapter weights are later placed under `Model-Pipeline/adapters/` (or provided via env var), the backend will automatically start using them without manual code changes**.
