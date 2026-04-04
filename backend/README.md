# FitSenseAI Backend

Local FastAPI backend for the FitSenseAI Flutter app.

## What is included

- auth (`/auth/signup`, `/auth/login`)
- onboarding/profile persistence (`/profile/onboarding`)
- plan generation and modification (`/plans`, `/plans/current`, `/plans/{plan_id}:modify`)
- workout logging (`/workouts`, `/workouts/{id}/exercises`, `/workouts/{id}/sets`)
- daily logs (`/daily/sleep`, `/daily/calories`, `/daily/weight`)
- dashboard aggregation (`/dashboard`)
- coaching (`/coach`, `/coach/stream`)
- next-week adaptation (`/adaptation:next_week`)

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open API docs at `http://127.0.0.1:8000/docs`.

## Reset local DB

```bash
python scripts/reset_db.py
```


## Student LLM auto-integration

This backend is patched to auto-discover a trained student adapter under `Model-Pipeline/adapters/`.
If a trained adapter exists and the optional inference dependencies are installed, the backend will automatically try to use the student model for:

- `/plans` plan generation
- `/plans/{plan_id}:modify` plan updates
- `/coach` and `/coach/stream`

If no trained adapter is present, or optional model dependencies are missing, the backend falls back to the existing rule-based logic so the app still runs.

### Optional model-serving dependencies

```bash
pip install -r requirements-llm.txt
```

### Optional environment variables

- `FITSENSE_STUDENT_ADAPTER_PATH` — explicit local adapter directory
- `FITSENSE_STUDENT_BASE_MODEL` — override the base HF model name
- `FITSENSE_STUDENT_REGISTRY_RECORD` — explicit registry record JSON path

### Runtime check

After login, call:

```text
GET /model/runtime
```

This returns whether the student LLM is available, which base model is configured, and which adapter path was discovered.


## Local training -> backend auto-pickup

Train the student model separately on your machine:

```bash
python3 -m pip install -r ../Model-Pipeline/requirements-local-train.txt
python3 ../Model-Pipeline/scripts/trainmodel.py
```

That writes the adapter under `Model-Pipeline/adapters/qwen3-8b-fitsense/<run_id>/` and a manifest file at
`Model-Pipeline/reports/latest_student_adapter.json`.

The backend runtime re-scans for that manifest and adapter automatically, so after training finishes you can restart the backend and it will use the newest adapter without editing code.


## Debugging behavior added

The backend now exposes clearer fallback information:

- `GET /model/runtime` shows whether a student adapter is truly runnable, not just whether a registry file exists.
- `POST /coach` includes `execution_debug` with the selected backend and fallback reason.
- `GET /coach/stream` sends one initial SSE event containing `debug` before token deltas.
- Background plan jobs now mention in their progress text whether the student model was used or why rules were used instead.


## Accepted student model layouts

The backend now accepts any of these:

1. LoRA adapter directory
   - `adapter_config.json`
   - `adapter_model.safetensors` or `adapter_model.bin`

2. Full merged model directory
   - `config.json`
   - model weights such as `*.safetensors` or `pytorch_model*.bin`

3. Artifact package
   - Set `FITSENSE_STUDENT_ARTIFACT` to a local directory, `.zip`, `.tar`, or `.tar.gz`
   - Or keep a registry `gcs_uri` and run with `gsutil` installed so the backend can download it automatically
