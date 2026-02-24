# FitSenseAI Information Architecture

Last updated: 2026-02-22

## 1. Purpose

Define the MVP information architecture across:

- client navigation and core screens,
- domain objects and their relationships,
- how information flows into plan generation, logging, and adaptation.

## 2. Primary Objects

Aligned to `database/tables.sql`:

- User identity and profile: `users`, `user_profiles`
- Goals and constraints: `goals`, `user_goals`, `conditions`, `user_conditions`, `user_medical_profiles`
- Plans: `workout_plans`, `plan_exercises`, `plan_sets`
- Executed workouts: `workouts`, `workout_exercises`, `workout_sets`
- Daily logs: `calorie_intake_logs`, `sleep_duration_logs`, `weight_logs`
- AI interactions: `ai_interactions`

## 3. Client IA (Screens)

MVP screens and navigation:

- Onboarding
- Home dashboard
- Current plan
- Workout session (logging)
- History (workouts and daily logs)
- Daily check-in (sleep, calories, weight)
- Coach (chat)
- Settings (equipment, constraints, targets)

## 4. Key User Flows

### 4.1 Onboarding -> Plan Creation

1. User enters goal + availability + equipment + constraints.
2. Backend persists user profile and goal linkage.
3. Backend requests plan generation from the student model.
4. Backend persists the plan and returns a structured representation to the client.

### 4.2 Plan -> Workout Logging

1. User starts a workout session from a plan day.
2. Client renders exercises and target sets.
3. User logs completed sets and notes.
4. Backend persists workout session, exercises, and sets.

### 4.3 Logs -> Adaptation

1. Backend aggregates the recent training period.
2. Backend computes adherence and simple trends.
3. Backend requests adaptation guidance from the student model, grounded in the aggregated summary.
4. Backend returns recommendations and optionally updates the plan.

### 4.4 Coaching

1. User asks a question.
2. Backend composes a short context summary (goal, constraints, recent workouts).
3. Backend calls the student model with streaming enabled.
4. Interaction is logged for auditing and quality monitoring.

## 5. Information Partitioning (Privacy)

- Online DB stores user-specific logs and plan data.
- Offline pipeline produces synthetic datasets for training and does not depend on real user data.
- AI interaction logging should avoid storing unnecessary sensitive content (store references or redacted summaries when possible).

