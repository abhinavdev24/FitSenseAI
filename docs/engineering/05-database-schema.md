# FitSenseAI Database Schema (MVP)

Last updated: 2026-02-22

Source of truth: `database/tables.sql`

## 1. Storage Choice

- Cloud SQL for PostgreSQL.
- UUID primary keys (already modeled).
- UTC timestamps for all event times.

## 2. Core Table Groups

### 2.1 Users and Profiles

- `users`
- `user_profiles`
- `user_medical_profiles`
- `user_medications`
- `user_allergies`

### 2.2 Goals and Conditions

- `goals`, `user_goals`
- `conditions`, `user_conditions`

### 2.3 Plans (Prescriptive)

- `workout_plans`
- `plan_exercises`
- `plan_sets`

### 2.4 Workouts (Descriptive)

- `workouts`
- `workout_exercises`
- `workout_sets`

### 2.5 Daily Logs

- `calorie_targets`, `calorie_intake_logs`
- `sleep_targets`, `sleep_duration_logs`
- `weight_logs`

### 2.6 AI Interactions

- `ai_interactions` for prompts/responses and model metadata.

## 3. Relationship Notes

- A user can have multiple plans; one is marked active (`is_active`).
- Workouts may be linked to a plan (`plan_id`) to support adherence and progression.
- Plan exercises and workout exercises both reference `exercises` for catalog consistency.

## 4. Recommended Indexes (MVP)

Add indexes during implementation:

- `workouts(user_id, started_at)`
- `workout_sets(workout_exercise_id, set_number)`
- `ai_interactions(user_id, created_at)`
- `weight_logs(user_id, logged_at)`
- `calorie_intake_logs(user_id, log_date)`
- `sleep_duration_logs(user_id, log_date)`

## 5. Migration Strategy

MVP expectation:

- Manage schema changes via migrations (tool choice deferred until backend framework is selected).
- Keep `database/tables.sql` in sync with migrations as a readable schema export.

