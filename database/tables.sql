CREATE TABLE "users" (
  "user_id" uuid PRIMARY KEY,
  "name" varchar,
  "email" varchar UNIQUE,
  "created_at" datetime
);

CREATE TABLE "goals" (
  "goal_id" uuid PRIMARY KEY,
  "name" varchar UNIQUE,
  "description" text
);

CREATE TABLE "user_goals" (
  "user_id" uuid NOT NULL,
  "goal_id" uuid NOT NULL,
  "priority" int
);

CREATE TABLE "conditions" (
  "condition_id" uuid PRIMARY KEY,
  "name" varchar UNIQUE,
  "description" text
);

CREATE TABLE "user_conditions" (
  "user_id" uuid NOT NULL,
  "condition_id" uuid NOT NULL,
  "severity" varchar,
  "notes" text
);

CREATE TABLE "equipment" (
  "equipment_id" uuid PRIMARY KEY,
  "name" varchar UNIQUE,
  "category" varchar
);

CREATE TABLE "exercises" (
  "exercise_id" uuid PRIMARY KEY,
  "name" varchar UNIQUE,
  "primary_muscle" varchar,
  "notes" text
);

CREATE TABLE "exercise_equipment" (
  "exercise_id" uuid NOT NULL,
  "equipment_id" uuid NOT NULL
);

CREATE TABLE "workout_plans" (
  "plan_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "name" varchar,
  "is_active" boolean,
  "created_at" datetime
);

CREATE TABLE "plan_exercises" (
  "plan_exercise_id" uuid PRIMARY KEY,
  "plan_id" uuid NOT NULL,
  "exercise_id" uuid NOT NULL,
  "position" int,
  "notes" text
);

CREATE TABLE "plan_sets" (
  "plan_set_id" uuid PRIMARY KEY,
  "plan_exercise_id" uuid NOT NULL,
  "set_number" int,
  "target_reps" int,
  "target_weight" decimal,
  "target_rir" int,
  "rest_seconds" int
);

CREATE TABLE "workouts" (
  "workout_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "plan_id" uuid,
  "started_at" datetime,
  "ended_at" datetime,
  "notes" text
);

CREATE TABLE "workout_exercises" (
  "workout_exercise_id" uuid PRIMARY KEY,
  "workout_id" uuid NOT NULL,
  "exercise_id" uuid NOT NULL,
  "plan_exercise_id" uuid,
  "position" int,
  "notes" text
);

CREATE TABLE "workout_sets" (
  "workout_set_id" uuid PRIMARY KEY,
  "workout_exercise_id" uuid NOT NULL,
  "set_number" int,
  "reps" int,
  "weight" decimal,
  "rir" int,
  "is_warmup" boolean,
  "completed_at" datetime
);

CREATE TABLE "ai_interactions" (
  "ai_interaction_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "context_type" varchar,
  "context_id" uuid,
  "query_text" text,
  "response_text" text,
  "model_name" varchar,
  "created_at" datetime
);

CREATE TABLE "user_profiles" (
  "user_id" uuid PRIMARY KEY,
  "date_of_birth" date,
  "sex" varchar,
  "height_cm" decimal,
  "activity_level" varchar,
  "updated_at" datetime
);

CREATE TABLE "user_medical_profiles" (
  "medical_profile_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "has_injuries" boolean,
  "injury_details" text,
  "surgeries_history" text,
  "family_history" text,
  "notes" text,
  "updated_at" datetime
);

CREATE TABLE "user_medications" (
  "medication_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "medication_name" varchar NOT NULL,
  "dosage" varchar,
  "frequency" varchar,
  "start_date" date,
  "end_date" date,
  "notes" text
);

CREATE TABLE "user_allergies" (
  "allergy_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "allergen" varchar NOT NULL,
  "reaction" varchar,
  "severity" varchar,
  "notes" text
);

CREATE TABLE "calorie_targets" (
  "calorie_target_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "maintenance_calories" int NOT NULL,
  "method" varchar,
  "effective_from" date NOT NULL,
  "effective_to" date,
  "notes" text,
  "created_at" datetime
);

CREATE TABLE "calorie_intake_logs" (
  "calorie_log_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "log_date" date NOT NULL,
  "calories_consumed" int NOT NULL,
  "notes" text,
  "created_at" datetime
);

CREATE TABLE "sleep_targets" (
  "sleep_target_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "target_sleep_hours" decimal NOT NULL,
  "effective_from" date NOT NULL,
  "effective_to" date,
  "notes" text,
  "created_at" datetime
);

CREATE TABLE "sleep_duration_logs" (
  "sleep_log_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "log_date" date NOT NULL,
  "sleep_duration_hours" decimal NOT NULL,
  "notes" text,
  "created_at" datetime
);

CREATE TABLE "weight_logs" (
  "weight_log_id" uuid PRIMARY KEY,
  "user_id" uuid NOT NULL,
  "logged_at" datetime NOT NULL,
  "weight_kg" decimal NOT NULL,
  "body_fat_percentage" decimal,
  "notes" text
);

CREATE UNIQUE INDEX ON "user_goals" ("user_id", "goal_id");

CREATE INDEX ON "user_goals" ("goal_id");

CREATE UNIQUE INDEX ON "user_conditions" ("user_id", "condition_id");

CREATE INDEX ON "user_conditions" ("condition_id");

CREATE UNIQUE INDEX ON "exercise_equipment" ("exercise_id", "equipment_id");

CREATE INDEX ON "exercise_equipment" ("equipment_id");

CREATE INDEX ON "workout_plans" ("user_id");

CREATE INDEX ON "workout_plans" ("user_id", "is_active");

CREATE INDEX ON "plan_exercises" ("plan_id");

CREATE INDEX ON "plan_exercises" ("exercise_id");

CREATE INDEX ON "plan_exercises" ("plan_id", "position");

CREATE INDEX ON "plan_sets" ("plan_exercise_id");

CREATE UNIQUE INDEX ON "plan_sets" ("plan_exercise_id", "set_number");

CREATE INDEX ON "workouts" ("user_id");

CREATE INDEX ON "workouts" ("plan_id");

CREATE INDEX ON "workouts" ("user_id", "started_at");

CREATE INDEX ON "workout_exercises" ("workout_id");

CREATE INDEX ON "workout_exercises" ("exercise_id");

CREATE INDEX ON "workout_exercises" ("plan_exercise_id");

CREATE INDEX ON "workout_exercises" ("workout_id", "position");

CREATE INDEX ON "workout_sets" ("workout_exercise_id");

CREATE UNIQUE INDEX ON "workout_sets" ("workout_exercise_id", "set_number");

CREATE INDEX ON "ai_interactions" ("user_id");

CREATE INDEX ON "ai_interactions" ("context_type", "context_id");

CREATE INDEX ON "ai_interactions" ("created_at");

CREATE INDEX ON "user_medical_profiles" ("user_id");

CREATE INDEX ON "user_medications" ("user_id");

CREATE INDEX ON "user_medications" ("medication_name");

CREATE INDEX ON "user_allergies" ("user_id");

CREATE INDEX ON "user_allergies" ("allergen");

CREATE INDEX ON "calorie_targets" ("user_id");

CREATE INDEX ON "calorie_targets" ("user_id", "effective_from");

CREATE INDEX ON "calorie_intake_logs" ("user_id");

CREATE UNIQUE INDEX ON "calorie_intake_logs" ("user_id", "log_date");

CREATE INDEX ON "sleep_targets" ("user_id");

CREATE INDEX ON "sleep_targets" ("user_id", "effective_from");

CREATE INDEX ON "sleep_duration_logs" ("user_id");

CREATE UNIQUE INDEX ON "sleep_duration_logs" ("user_id", "log_date");

CREATE INDEX ON "weight_logs" ("user_id");

CREATE INDEX ON "weight_logs" ("user_id", "logged_at");

ALTER TABLE "user_goals" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "user_goals" ADD FOREIGN KEY ("goal_id") REFERENCES "goals" ("goal_id");

ALTER TABLE "user_conditions" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "user_conditions" ADD FOREIGN KEY ("condition_id") REFERENCES "conditions" ("condition_id");

ALTER TABLE "exercise_equipment" ADD FOREIGN KEY ("exercise_id") REFERENCES "exercises" ("exercise_id");

ALTER TABLE "exercise_equipment" ADD FOREIGN KEY ("equipment_id") REFERENCES "equipment" ("equipment_id");

ALTER TABLE "workout_plans" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "plan_exercises" ADD FOREIGN KEY ("plan_id") REFERENCES "workout_plans" ("plan_id");

ALTER TABLE "plan_exercises" ADD FOREIGN KEY ("exercise_id") REFERENCES "exercises" ("exercise_id");

ALTER TABLE "plan_sets" ADD FOREIGN KEY ("plan_exercise_id") REFERENCES "plan_exercises" ("plan_exercise_id");

ALTER TABLE "workouts" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "workouts" ADD FOREIGN KEY ("plan_id") REFERENCES "workout_plans" ("plan_id");

ALTER TABLE "workout_exercises" ADD FOREIGN KEY ("workout_id") REFERENCES "workouts" ("workout_id");

ALTER TABLE "workout_exercises" ADD FOREIGN KEY ("exercise_id") REFERENCES "exercises" ("exercise_id");

ALTER TABLE "workout_exercises" ADD FOREIGN KEY ("plan_exercise_id") REFERENCES "plan_exercises" ("plan_exercise_id");

ALTER TABLE "workout_sets" ADD FOREIGN KEY ("workout_exercise_id") REFERENCES "workout_exercises" ("workout_exercise_id");

ALTER TABLE "ai_interactions" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "user_profiles" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "user_medical_profiles" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "user_medications" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "user_allergies" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "calorie_targets" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "calorie_intake_logs" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "sleep_targets" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "sleep_duration_logs" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");

ALTER TABLE "weight_logs" ADD FOREIGN KEY ("user_id") REFERENCES "users" ("user_id");
