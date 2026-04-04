# FitSenseAI User Stories and Acceptance Criteria

Last updated: 2026-02-22

## 1. Onboarding and Profile

### Story: Capture user goal and constraints

As a user, I want to enter my goal, schedule, equipment, and constraints so the plan is personalized.

Acceptance criteria:

- Goal type is captured (fat loss, muscle gain, strength, general fitness).
- Days/week and equipment availability are captured.
- Injury/constraint flags can be captured at a high level.
- The backend persists the profile and goal linkage in the database.

## 2. Plan Generation

### Story: Generate a weekly plan

As a user, I want a weekly plan generated from my goal and constraints so I know what to do each day.

Acceptance criteria:

- A structured weekly plan is returned with at least:
  - session schedule,
  - exercise list per session,
  - sets/reps/target RIR and rest.
- A short explanation is provided (progression and safety notes).
- The plan is persisted (`workout_plans`, `plan_exercises`, `plan_sets`).

## 3. Plan Modification

### Story: Swap exercises

As a user, I want to swap an exercise to one I prefer or can perform safely.

Acceptance criteria:

- User can request a swap with a reason (equipment, pain, preference).
- Updated plan maintains workout structure and progression intent.
- Contraindicated exercises are avoided when injury flags exist.

### Story: Change schedule days

As a user, I want to change the number of training days so the plan fits my week.

Acceptance criteria:

- Plan is restructured to match requested days/week.
- Total workload is adjusted proportionally and explained.

## 4. Workout Logging

### Story: Log a workout session

As a user, I want to log sets/reps/weight/RIR so the app can adapt my plan.

Acceptance criteria:

- User can create a workout session and attach it to a plan (optional).
- User can log exercises and sets with reps/weight/RIR and a note.
- Data persists in `workouts`, `workout_exercises`, `workout_sets`.

## 5. Daily Logs

### Story: Log sleep

As a user, I want to log sleep duration so recovery can be considered.

Acceptance criteria:

- User can submit sleep duration for a date.
- Data persists in `sleep_duration_logs`.

### Story: Log calories

As a user, I want to log total calorie intake so nutrition context can be considered.

Acceptance criteria:

- User can submit total calories for a date.
- Data persists in `calorie_intake_logs`.

### Story: Log weight

As a user, I want to log my weight over time to track progress.

Acceptance criteria:

- User can submit weight at a timestamp.
- Data persists in `weight_logs`.

## 6. Adaptation and Coaching

### Story: Get next-week adjustments

As a user, I want guidance based on my recent training so my plan adapts.

Acceptance criteria:

- Backend uses recent workouts and logs to produce an adjustment summary.
- Response explains what changed and why (volume/intensity/schedule).

### Story: Ask coaching questions

As a user, I want to ask questions and get concise guidance.

Acceptance criteria:

- The system responds within the scoped fitness domain.
- The response streams progressively.
- Interaction is logged to `ai_interactions` with model name/version.

### Story: Safety-aware behavior for injury/pain prompts

As a user, I want safe guidance when I mention pain or injury.

Acceptance criteria:

- Response is conservative and includes escalation language.
- Unsafe instructions (ignore pain, max out constantly) are not produced.
- Safety-related metadata is available for auditing (offline and online).

## 7. Reliability and Limits

### Story: Handle overload gracefully

As a user, I want the app to behave predictably when many requests happen at once.

Acceptance criteria:

- Backend queues LLM requests or returns `429` with retry guidance.
- Maximum output tokens and timeouts are enforced.

