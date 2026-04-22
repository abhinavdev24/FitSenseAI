# FitSenseAI Backend

FastAPI backend for the FitSenseAI fitness coaching application.

[![API Docs](https://img.shields.io/badge/API%20Docs-fitsense--backend.abhinavdev24.com-2563EB?style=for-the-badge&logo=fastapi&logoColor=white)](https://fitsense-backend.abhinavdev24.com/docs)
[![ER Diagram](https://img.shields.io/badge/MySQL-ER%20Diagram-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://dbdiagram.io/d/FitSenseAI-69850002bd82f5fce2cfe02c)

## Endpoints

### Auth & Profile

- `POST /auth/signup` — create account
- `POST /auth/login` — login, returns bearer token
- `GET /me` — current user profile
- `POST /profile/onboarding` — save onboarding data (age, sex, goals, equipment, medical info)

### Plans

- `POST /plans` — generate a new workout plan (async background job)
- `GET /plans/current` — get the active plan with all days/exercises/sets
- `POST /plans/{plan_id}:modify` — modify the active plan with a natural language instruction
- `GET /plans/jobs/{job_id}` — poll plan generation job status
- `GET /plans/jobs/latest` — latest pending job
- `POST /pipeline/trigger` — alias for plan generation

### Workouts

- `POST /workouts` — start a new workout session
- `POST /workouts/{id}/exercises` — log an exercise in a workout
- `POST /workouts/{id}/sets` — log a set
- `GET /workouts/recent` — recent workout summaries

### Daily Logs

- `POST /daily/sleep` — log sleep hours
- `POST /daily/calories` — log calorie intake
- `POST /daily/weight` — log body weight

### Targets

- `POST /targets/calories` — set a calorie target
- `GET /targets/calories` — list calorie targets
- `POST /targets/sleep` — set a sleep target
- `GET /targets/sleep` — list sleep targets

### Coaching

- `POST /coach` — ask the AI coach a question
- `GET /coach/stream` — SSE streaming version of coach
- `POST /adaptation:next_week` — get next-week training adaptation suggestions

### Other

- `GET /catalog/exercises` — list all exercises in the database
- `GET /model/runtime` — student LLM runtime status
- `GET /dashboard` — aggregated profile, plan, workouts, and logs

## Architecture

### System Overview

```mermaid
flowchart TD
    subgraph Client["📱 Client"]
        style Client fill:#dbeafe,color:#1e3a5f,stroke:#2563EB
        MA[Mobile App]
    end

    subgraph Backend["⚙️ FastAPI Backend"]
        style Backend fill:#d1fae5,color:#14532d,stroke:#16A34A
        API[API Routes\nmain.py]
        SVC[Services\nservices.py]
        LLM[LLM Runtime\nllm_runtime.py]
        API --> SVC
        SVC --> LLM
    end

    subgraph DB["🗄️ Database"]
        style DB fill:#ede9fe,color:#3b0764,stroke:#9333EA
        MySQL[(Cloud SQL\nMySQL\nproduction)]
        SQLite[(SQLite\nlocal dev)]
    end

    subgraph LLMProviders["🤖 LLM Providers"]
        style LLMProviders fill:#fef3c7,color:#7c2d12,stroke:#EA580C
        P1["① OpenAI-compatible API\nGroq · OpenRouter · Together"]
        P2["② Cloud Run\nvLLM service"]
        P3["③ Local LoRA Adapter\nStudent model on-device"]
        P4["④ Rule-Based Fallback\nAlways available"]
    end

    MA -->|"Bearer Token · HTTPS"| API
    SVC -->|"SQLAlchemy ORM"| MySQL
    SVC -->|"SQLAlchemy ORM"| SQLite
    LLM -->|"Priority 1"| P1
    LLM -->|"Priority 2"| P2
    LLM -->|"Priority 3"| P3
    LLM -->|"Priority 4"| P4
```

### Plan Generation Flow

```mermaid
sequenceDiagram
    autonumber
    participant C as 📱 Client
    participant API as ⚙️ FastAPI
    participant DB as 🗄️ Database
    participant BG as 🔄 Background Task
    participant LLM as 🤖 LLM Runtime

    C->>API: POST /plans
    API->>DB: INSERT PlanGenerationJob (status=pending)
    API-->>C: { job_id, status: "pending" }
    API-)BG: Spawn background task

    loop Poll for completion
        C->>API: GET /plans/jobs/{job_id}
        API-->>C: { status, progress_message }
    end

    BG->>DB: UPDATE job → status=running
    BG->>LLM: try_student_plan_generation()

    alt LLM succeeds
        LLM-->>BG: Validated JSON plan
        BG->>DB: INSERT WorkoutPlan + PlanDays + PlanExercises + PlanSets
        BG->>DB: UPDATE job → status=completed
    else LLM unavailable or invalid JSON
        LLM-->>BG: Error / None
        BG->>BG: generate_plan() — rule-based split selection
        BG->>DB: INSERT rule-based WorkoutPlan
        BG->>DB: UPDATE job → status=completed
    end

    C->>API: GET /plans/jobs/{job_id}
    API-->>C: { status: "completed", result_plan: { ... } }
```

### Request Authentication

```mermaid
sequenceDiagram
    autonumber
    participant C as 📱 Client
    participant API as ⚙️ FastAPI
    participant DB as 🗄️ Database

    C->>API: Any request + Authorization: Bearer <token>
    API->>DB: SELECT SessionToken WHERE token = ?

    alt Token not found or expired
        DB-->>API: None
        API-->>C: 401 Unauthorized
    else Token valid
        DB-->>API: SessionToken { user_id }
        Note over API,DB: Eager load: profile, preference,<br/>medical_profile, goals, conditions,<br/>plans, sessions
        API->>DB: SELECT User + relationships
        DB-->>API: Hydrated User object
        API->>API: Inject into route handler
        API-->>C: 200 + response payload
    end
```

### AI Coaching Flow

```mermaid
flowchart TD
    subgraph Input["Request"]
        style Input fill:#dbeafe,color:#1e3a5f,stroke:#2563EB
        REQ["POST /coach\nor GET /coach/stream (SSE)"]
    end

    subgraph Context["Context Assembly"]
        style Context fill:#d1fae5,color:#14532d,stroke:#16A34A
        CTX["User profile + goals + conditions\nRecent workouts & sets\nSleep · calorie · weight logs"]
    end

    subgraph Inference["Inference"]
        style Inference fill:#fef3c7,color:#7c2d12,stroke:#EA580C
        STU["try_student_coach_reply()\nLLM Runtime"]
        RUL["build_coach_reply()\nRule-Based"]
    end

    subgraph RuleChecks["Rule-Based Checks"]
        style RuleChecks fill:#ede9fe,color:#3b0764,stroke:#9333EA
        PAIN["Pain / injury keywords?\n→ Recommend rest"]
        LOG["Low logging volume?\n→ Encourage tracking"]
        SLP["Sleep below target?\n→ Recovery note"]
    end

    subgraph Output["Output"]
        style Output fill:#ccfbf1,color:#134e4a,stroke:#0D9488
        LOGAI["INSERT AIInteraction\nquery · response · model_name · context_type"]
        RESP["Return text response\nor SSE token stream"]
    end

    REQ --> CTX
    CTX --> STU
    STU -->|LLM available| LOGAI
    STU -->|LLM unavailable| RUL
    RUL --> PAIN --> LOG --> SLP --> LOGAI
    LOGAI --> RESP
```

## Setup

```bash
cd backend
cp .env.example .env.local   # edit with your values
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs at `http://127.0.0.1:8000/docs`.

## Database

Set `DATABASE_ENGINE` in `.env.local`. Supported: `mysql`, `sqlite`.

**MySQL (Cloud SQL):**

```env
DATABASE_ENGINE=mysql
DATABASE_USER=fitsensebackend
DATABASE_PASSWORD=your_password_here
DATABASE_NAME=fitsense
DATABASE_HOST=35.224.89.210
DATABASE_PORT=3306
```

**SQLite (local dev):**

```env
DATABASE_ENGINE=sqlite
DATABASE_PATH=/absolute/path/to/fitsense.db
```

Tables are created automatically on startup via `Base.metadata.create_all`.

Reset the database:

```bash
python scripts/reset_db.py
```

## LLM Inference

The backend uses an OpenAI-compatible chat completions API for plan generation and coaching. Configure in `.env.local`:

```env
OPENAI_API_KEY=your_api_key
OPENAI_API_URL=https://your-provider/v1/chat/completions
OPENAI_MODEL=your-model-id
MAX_OUTPUT_TOKENS=13000
```

Any provider exposing a `/v1/chat/completions` endpoint works (Groq, Together AI, Ollama, LM Studio, etc).

If no API is configured, the backend falls back to rule-based plan generation and coaching.

### Inference priority

1. **OpenAI-compatible API** — if `OPENAI_API_KEY` and `OPENAI_API_URL` are set
2. **Cloud Run** — if `FITSENSE_CLOUDRUN_URL` is set (deployed vLLM service)
3. **Local model** — if a trained LoRA adapter is discovered and inference dependencies are installed
4. **Rule-based fallback** — always available

### Student model auto-discovery

The backend scans `Model-Pipeline/adapters/` for trained LoRA adapters. If found (and optional deps installed), it uses the student model directly.

Accepted layouts:

- **LoRA adapter**: `adapter_config.json` + `adapter_model.safetensors`
- **Full merged model**: `config.json` + model weights
- **Artifact package**: set `FITSENSE_STUDENT_ARTIFACT` to a local `.zip`/`.tar.gz` or directory

Optional env vars:

- `FITSENSE_STUDENT_ADAPTER_PATH` — explicit adapter directory
- `FITSENSE_STUDENT_BASE_MODEL` — override base model name
- `FITSENSE_STUDENT_REGISTRY_RECORD` — explicit registry record JSON

Install optional model-serving dependencies:

```bash
pip install -r requirements-llm.txt
```

### Runtime check

```
GET /model/runtime
```

Returns whether the student LLM is available, the base model, adapter path, and provider details.

## AI Interaction Logging

All LLM calls are logged to the `ai_interactions` table with:

- `context_type`: route that triggered the call (`coach`, `coach-stream`, `plan-generate`, `plan-modify`, or `failed`)
- `query_text`: full user prompt sent to the LLM
- `response_text`: full raw LLM response
- `model_name`: model used for inference

Failed LLM calls are logged with `context_type="failed"` and the error in `response_text`.

## Debugging

- `GET /model/runtime` — shows whether the student adapter is runnable
- `POST /coach` response includes `execution_debug` with selected backend and fallback reason
- `GET /coach/stream` sends an initial SSE event with `debug` before token deltas
- Plan jobs include progress text indicating whether the student model or rules were used
- Set `FITSENSE_DEBUG_VERTEX=1` in `.env.local` for verbose inference logging
