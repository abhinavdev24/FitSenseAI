# FitSenseAI

AI-powered fitness coaching application focused on personalized workouts, progress tracking, and health-aware guidance.

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

## Roadmap Snapshot

- Phase 1: Foundation and data schemas
- Phase 2: Model development (teacher/student workflow)
- Phase 3: Backend + app MVP
- Phase 4: Adaptation engine and instrumentation
- Phase 5: Safety, monitoring, hardening
- Phase 6: Pilot, iteration, final validation

## Notes

- This repository currently contains planning and database-design artifacts.
- As services are added, extend this README with setup commands, environment variables, and deployment instructions.
