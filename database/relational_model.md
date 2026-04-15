# Relational Model — FitSense AI

Primary keys are underlined. Foreign keys are marked with *.

---

## Users & Goals

users(<u>user_id</u>, name, email, created_at)

goals(<u>goal_id</u>, name, description)

user_goals(<u>user_id</u>, <u>goal_id</u>, priority)

---

## Conditions

conditions(<u>condition_id</u>, name, description)

user_conditions(<u>user_id</u>, <u>condition_id</u>, severity, notes)

---

## Equipment & Exercises

equipment(<u>equipment_id</u>, name, category)

exercises(<u>exercise_id</u>, name, primary_muscle, notes, video_url, thumbnail_base64)

exercise_equipment(<u>exercise_id</u>, <u>equipment_id</u>)

---

## Workout Plans (Blueprint)

workout_plans(<u>plan_id</u>, user_id, name, is_active, created_at)

plan_days(<u>plan_day_id</u>, plan_id, name, day_order, notes)

plan_exercises(<u>plan_exercise_id</u>, plan_day_id, exercise_id, position, notes)

plan_sets(<u>plan_set_id</u>, plan_exercise_id, set_number, target_reps, target_weight, target_rir, rest_seconds)

---

## Workouts (Actual Sessions)

workouts(<u>workout_id</u>, user_id, plan_id, plan_day_id, started_at, ended_at, notes)

workout_exercises(<u>workout_exercise_id</u>, workout_id, exercise_id, plan_exercise_id, position, notes)

workout_sets(<u>workout_set_id</u>, workout_exercise_id, set_number, reps, weight, rir, is_warmup, completed_at)

---

## AI Interactions

ai_interactions(<u>ai_interaction_id</u>, user_id, context_type, context_id, query_text, response_text, model_name, created_at)

---

## User Profiles & Medical

user_profiles(<u>user_id</u>, date_of_birth, sex, height_cm, activity_level, updated_at)

user_medical_profiles(<u>medical_profile_id</u>, user_id, has_injuries, injury_details, surgeries_history, family_history, notes, updated_at)

user_medications(<u>medication_id</u>, user_id, medication_name, dosage, frequency, start_date, end_date, notes)

user_allergies(<u>allergy_id</u>, user_id, allergen, reaction, severity, notes)

---

## Calorie Tracking

calorie_targets(<u>calorie_target_id</u>, user_id, maintenance_calories, method, effective_from, effective_to, notes, created_at)

calorie_intake_logs(<u>calorie_log_id</u>, user_id, log_date, calories_consumed, notes, created_at)

---

## Sleep Tracking

sleep_targets(<u>sleep_target_id</u>, user_id, target_sleep_hours, effective_from, effective_to, notes, created_at)

sleep_duration_logs(<u>sleep_log_id</u>, user_id, log_date, sleep_duration_hours, notes, created_at)

---

## Weight Tracking

weight_logs(<u>weight_log_id</u>, user_id, logged_at, weight_kg, body_fat_percentage, notes)

---

## Foreign Keys

user_goals.user_id -> users.user_id (NOT NULL)
user_goals.goal_id -> goals.goal_id (NOT NULL)

user_conditions.user_id -> users.user_id (NOT NULL)
user_conditions.condition_id -> conditions.condition_id (NOT NULL)

exercise_equipment.exercise_id -> exercises.exercise_id (NOT NULL)
exercise_equipment.equipment_id -> equipment.equipment_id (NOT NULL)

workout_plans.user_id -> users.user_id (NOT NULL)

plan_days.plan_id -> workout_plans.plan_id (NOT NULL)

plan_exercises.plan_day_id -> plan_days.plan_day_id (NOT NULL)
plan_exercises.exercise_id -> exercises.exercise_id (NOT NULL)

plan_sets.plan_exercise_id -> plan_exercises.plan_exercise_id (NOT NULL)

workouts.user_id -> users.user_id (NOT NULL)
workouts.plan_id -> workout_plans.plan_id (NULL ALLOWED)
workouts.plan_day_id -> plan_days.plan_day_id (NULL ALLOWED)

workout_exercises.workout_id -> workouts.workout_id (NOT NULL)
workout_exercises.exercise_id -> exercises.exercise_id (NOT NULL)
workout_exercises.plan_exercise_id -> plan_exercises.plan_exercise_id (NULL ALLOWED)

workout_sets.workout_exercise_id -> workout_exercises.workout_exercise_id (NOT NULL)

ai_interactions.user_id -> users.user_id (NOT NULL)

user_profiles.user_id -> users.user_id (NOT NULL)

user_medical_profiles.user_id -> users.user_id (NOT NULL)

user_medications.user_id -> users.user_id (NOT NULL)

user_allergies.user_id -> users.user_id (NOT NULL)

calorie_targets.user_id -> users.user_id (NOT NULL)

calorie_intake_logs.user_id -> users.user_id (NOT NULL)

sleep_targets.user_id -> users.user_id (NOT NULL)

sleep_duration_logs.user_id -> users.user_id (NOT NULL)

weight_logs.user_id -> users.user_id (NOT NULL)

---

## Notes

- `user_goals`, `user_conditions`, and `exercise_equipment` use **composite primary keys**.
- `plan_days` is an intermediate table introduced between `workout_plans` and `plan_exercises` to support named day cycles (e.g. `PUSH_1`, `PULL_2`, `UPPER_1`) within a plan.
- `workouts.plan_id` and `workouts.plan_day_id` are both nullable — a session may be freeform with no plan, or follow a plan without being tied to a specific named day.
- `workout_exercises.plan_exercise_id` is nullable — it links an actual set back to its blueprint row when following a plan, but is omitted for ad-hoc exercises.
- `calorie_targets.effective_to` and `sleep_targets.effective_to` are nullable — a NULL value indicates the currently active target.
- `ai_interactions.context_id` is a polymorphic reference — its meaning depends on `context_type` (`plan`, `workout`, or `general`).
- `exercises.thumbnail_base64` stores a base64-encoded image directly in the database. For large deployments, consider offloading to object storage (S3/CDN) and storing only the URL, similar to `video_url`.
