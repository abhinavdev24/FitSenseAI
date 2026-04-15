
# Local student training and backend auto-pickup

This repository now supports a clean split:

- local student-model training happens in `Model-Pipeline/`
- backend inference stays in `backend/`
- the backend auto-discovers the newest trained adapter through `Model-Pipeline/reports/latest_student_adapter.json`

## Train locally

```bash
python3 -m pip install -r Model-Pipeline/requirements-local-train.txt
python3 Model-Pipeline/scripts/trainmodel.py
```

Optional flags:

```bash
python3 Model-Pipeline/scripts/trainmodel.py       --dataset-path Model-Pipeline/data/formatted/<run_id>/train_formatted.jsonl       --max-steps 100       --run-id my-local-run
```

## What training writes

- adapter folder: `Model-Pipeline/adapters/qwen3-8b-fitsense/<run_id>/`
- runtime manifest: `Model-Pipeline/reports/latest_student_adapter.json`
- registry-style record: `Model-Pipeline/reports/registry_record_<run_id>.json`

## How backend uses it

`backend/app/llm_runtime.py` now checks for:

1. `FITSENSE_STUDENT_ADAPTER_PATH`
2. `Model-Pipeline/reports/latest_student_adapter.json`
3. newest adapter under `Model-Pipeline/adapters/**/adapter_config.json`

The backend uses the adapter for:

- plan creation
- plan modification
- coach replies

If model loading fails, it falls back to the rule engine and prints a fallback log.
