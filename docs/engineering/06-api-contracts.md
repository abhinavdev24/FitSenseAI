# FitSenseAI API Contracts (MVP)

Last updated: 2026-02-22

This document defines the backend API contracts and the internal model serving contract.

## 1. Conventions

- JSON request/response bodies.
- All timestamps in ISO-8601 UTC.
- Auth mechanism is MVP-flexible, but endpoints must enforce per-user isolation.

## 2. Backend API (Public to Client)

### 2.1 Plans

- `POST /plans`
  - Request: goal + constraints overrides (optional)
  - Response: plan object + explanation text
- `POST /plans/{plan_id}:modify`
  - Request: modification instruction + optional constraints
  - Response: updated plan + explanation
- `GET /plans/current`
  - Response: active plan

### 2.2 Workouts (Logging)

- `POST /workouts`
  - Request: `plan_id` (optional), started_at
  - Response: workout_id
- `POST /workouts/{workout_id}/exercises`
  - Request: exercise_id, position, notes
  - Response: workout_exercise_id
- `POST /workouts/{workout_id}/sets`
  - Request: workout_exercise_id, set_number, reps, weight, rir, is_warmup, completed_at
  - Response: workout_set_id

### 2.3 Daily Logs

- `POST /daily/sleep`
- `POST /daily/calories`
- `POST /daily/weight`

Each accepts a date/time and a numeric value and returns the created row id.

### 2.4 Coaching (Streaming)

- `POST /coach`
  - Request: `message`, optional `context_mode`
  - Response: streaming text (SSE recommended)

### 2.5 Adaptation

- `POST /adaptation:next_week`
  - Request: optional time window
  - Response: adjustment summary and optional structured plan changes

## 3. Error Model (Backend)

- `400`: invalid request
- `401/403`: auth or authorization failure
- `404`: missing entity
- `409`: conflict (duplicate or state mismatch)
- `429`: rate limited / saturated
- `500`: internal error

## 4. Internal Inference API (Backend -> vLLM)

Use vLLM OpenAI-compatible API on a private network:

- `POST /v1/chat/completions`
  - `stream=true` for streaming token deltas
  - enforce `max_tokens` and context limits from the backend

Backend must treat the inference endpoint as an internal dependency and apply:

- timeouts,
- circuit breaking on repeated failures,
- a bounded queue for load spikes.

