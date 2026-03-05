-- ============================================================
-- FitSense AI — PostgreSQL Schema
-- Generated from database_design.dbml
-- Optimizations:
--   • Native UUID type (128-bit, indexed via B-tree by default)
--   • gen_random_uuid() for default PK generation (pgcrypto-free in PG 13+)
--   • SMALLINT / SMALLSERIAL for small-range numerics
--   • NUMERIC(n,s) for exact decimals
--   • CHECK constraints for ENUMs (more flexible than PG ENUM type)
--   • BOOLEAN native type
--   • TEXT instead of VARCHAR where no length limit is needed
--   • TIMESTAMPTZ (timezone-aware) for all timestamps
--   • Partial indexes for sparse/filtered queries
--   • ON DELETE CASCADE / SET NULL / RESTRICT on all FKs
--   • Indexes defined separately for clarity
-- ============================================================

-- ============================================================
-- Users & Auth
-- ============================================================

CREATE TABLE users (
  user_id    UUID         NOT NULL DEFAULT gen_random_uuid(),
  name       VARCHAR(120) NOT NULL,
  email      VARCHAR(254) NOT NULL,
  created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id),
  UNIQUE (email)
);

-- ============================================================
-- Goals
-- ============================================================

CREATE TABLE goals (
  goal_id     UUID         NOT NULL DEFAULT gen_random_uuid(),
  name        VARCHAR(120) NOT NULL,
  description TEXT,
  PRIMARY KEY (goal_id),
  UNIQUE (name)
);

CREATE TABLE user_goals (
  user_id  UUID NOT NULL,
  goal_id  UUID NOT NULL,
  -- lower number = higher priority
  priority SMALLINT CHECK (priority >= 0),
  PRIMARY KEY (user_id, goal_id),
  FOREIGN KEY (user_id) REFERENCES users  (user_id) ON DELETE CASCADE,
  FOREIGN KEY (goal_id) REFERENCES goals  (goal_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_goals_goal_id ON user_goals (goal_id);

-- ============================================================
-- Conditions
-- ============================================================

CREATE TABLE conditions (
  condition_id UUID         NOT NULL DEFAULT gen_random_uuid(),
  name         VARCHAR(120) NOT NULL,
  description  TEXT,
  PRIMARY KEY (condition_id),
  UNIQUE (name)
);

CREATE TABLE user_conditions (
  user_id      UUID NOT NULL,
  condition_id UUID NOT NULL,
  severity     VARCHAR(20) CHECK (severity IN ('mild', 'moderate', 'severe')),
  notes        TEXT,
  PRIMARY KEY (user_id, condition_id),
  FOREIGN KEY (user_id)      REFERENCES users      (user_id)      ON DELETE CASCADE,
  FOREIGN KEY (condition_id) REFERENCES conditions (condition_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_conditions_condition_id ON user_conditions (condition_id);

-- ============================================================
-- Equipment & Exercises
-- ============================================================

CREATE TABLE equipment (
  equipment_id UUID        NOT NULL DEFAULT gen_random_uuid(),
  name         VARCHAR(120) NOT NULL,
  category     VARCHAR(60),
  PRIMARY KEY (equipment_id),
  UNIQUE (name)
);

CREATE TABLE exercises (
  exercise_id      UUID         NOT NULL DEFAULT gen_random_uuid(),
  name             VARCHAR(150) NOT NULL,
  primary_muscle   VARCHAR(80),
  notes            TEXT,
  -- URL to hosted video file (e.g. S3/CDN)
  video_url        VARCHAR(500),
  -- base64-encoded thumbnail; TEXT in PG has no practical size limit
  thumbnail_base64 TEXT,
  PRIMARY KEY (exercise_id),
  UNIQUE (name)
);

CREATE TABLE exercise_equipment (
  exercise_id  UUID NOT NULL,
  equipment_id UUID NOT NULL,
  PRIMARY KEY (exercise_id, equipment_id),
  FOREIGN KEY (exercise_id)  REFERENCES exercises  (exercise_id)  ON DELETE CASCADE,
  FOREIGN KEY (equipment_id) REFERENCES equipment  (equipment_id) ON DELETE CASCADE
);

CREATE INDEX idx_exercise_equipment_equipment_id ON exercise_equipment (equipment_id);

-- ============================================================
-- Workout Plans (Blueprint)
-- ============================================================

CREATE TABLE workout_plans (
  plan_id    UUID         NOT NULL DEFAULT gen_random_uuid(),
  user_id    UUID         NOT NULL,
  name       VARCHAR(150),
  is_active  BOOLEAN      NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  PRIMARY KEY (plan_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_workout_plans_user_id ON workout_plans (user_id);
-- Partial index: quickly find the single active plan per user
CREATE INDEX idx_workout_plans_user_id_active
  ON workout_plans (user_id)
  WHERE is_active = TRUE;

-- Named days within a plan cycle (e.g. PUSH_1, PULL_2, UPPER_1)
CREATE TABLE plan_days (
  plan_day_id UUID        NOT NULL DEFAULT gen_random_uuid(),
  plan_id     UUID        NOT NULL,
  name        VARCHAR(60) NOT NULL,
  -- ordering within the plan cycle
  day_order   SMALLINT    CHECK (day_order >= 0),
  notes       TEXT,
  PRIMARY KEY (plan_day_id),
  UNIQUE (plan_id, name),
  FOREIGN KEY (plan_id) REFERENCES workout_plans (plan_id) ON DELETE CASCADE
);

CREATE INDEX idx_plan_days_plan_id           ON plan_days (plan_id);
CREATE INDEX idx_plan_days_plan_id_day_order ON plan_days (plan_id, day_order);

CREATE TABLE plan_exercises (
  plan_exercise_id UUID     NOT NULL DEFAULT gen_random_uuid(),
  -- references named day, not the plan directly
  plan_day_id      UUID     NOT NULL,
  exercise_id      UUID     NOT NULL,
  position         SMALLINT CHECK (position >= 0),
  notes            TEXT,
  PRIMARY KEY (plan_exercise_id),
  FOREIGN KEY (plan_day_id)  REFERENCES plan_days  (plan_day_id) ON DELETE CASCADE,
  FOREIGN KEY (exercise_id)  REFERENCES exercises  (exercise_id) ON DELETE RESTRICT
);

CREATE INDEX idx_plan_exercises_plan_day_id          ON plan_exercises (plan_day_id);
CREATE INDEX idx_plan_exercises_exercise_id          ON plan_exercises (exercise_id);
CREATE INDEX idx_plan_exercises_plan_day_id_position ON plan_exercises (plan_day_id, position);

CREATE TABLE plan_sets (
  plan_set_id      UUID     NOT NULL DEFAULT gen_random_uuid(),
  plan_exercise_id UUID     NOT NULL,
  set_number       SMALLINT CHECK (set_number > 0),
  target_reps      SMALLINT CHECK (target_reps >= 0),
  -- e.g. 0–999.99 kg
  target_weight    NUMERIC(6,2) CHECK (target_weight >= 0),
  -- reps-in-reserve: typically 0–10
  target_rir       SMALLINT CHECK (target_rir >= 0 AND target_rir <= 10),
  rest_seconds     SMALLINT CHECK (rest_seconds >= 0),
  PRIMARY KEY (plan_set_id),
  UNIQUE (plan_exercise_id, set_number),
  FOREIGN KEY (plan_exercise_id) REFERENCES plan_exercises (plan_exercise_id) ON DELETE CASCADE
);

CREATE INDEX idx_plan_sets_plan_exercise_id ON plan_sets (plan_exercise_id);

-- ============================================================
-- Workouts (Actual sessions)
-- ============================================================

CREATE TABLE workouts (
  workout_id  UUID        NOT NULL DEFAULT gen_random_uuid(),
  user_id     UUID        NOT NULL,
  -- nullable: freeform workout may not follow a plan
  plan_id     UUID        DEFAULT NULL,
  -- nullable: which named day template was followed (e.g. "PUSH_1")
  plan_day_id UUID        DEFAULT NULL,
  started_at  TIMESTAMPTZ,
  ended_at    TIMESTAMPTZ,
  notes       TEXT,
  -- ensure ended_at is after started_at when both are set
  CHECK (ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at),
  PRIMARY KEY (workout_id),
  FOREIGN KEY (user_id)     REFERENCES users         (user_id)     ON DELETE CASCADE,
  FOREIGN KEY (plan_id)     REFERENCES workout_plans (plan_id)     ON DELETE SET NULL,
  FOREIGN KEY (plan_day_id) REFERENCES plan_days     (plan_day_id) ON DELETE SET NULL
);

CREATE INDEX idx_workouts_user_id            ON workouts (user_id);
CREATE INDEX idx_workouts_plan_id            ON workouts (plan_id);
CREATE INDEX idx_workouts_plan_day_id        ON workouts (plan_day_id);
CREATE INDEX idx_workouts_user_id_started_at ON workouts (user_id, started_at DESC);

CREATE TABLE workout_exercises (
  workout_exercise_id UUID     NOT NULL DEFAULT gen_random_uuid(),
  workout_id          UUID     NOT NULL,
  exercise_id         UUID     NOT NULL,
  -- nullable link to the blueprint row
  plan_exercise_id    UUID     DEFAULT NULL,
  position            SMALLINT CHECK (position >= 0),
  notes               TEXT,
  PRIMARY KEY (workout_exercise_id),
  FOREIGN KEY (workout_id)       REFERENCES workouts       (workout_id)       ON DELETE CASCADE,
  FOREIGN KEY (exercise_id)      REFERENCES exercises      (exercise_id)      ON DELETE RESTRICT,
  FOREIGN KEY (plan_exercise_id) REFERENCES plan_exercises (plan_exercise_id) ON DELETE SET NULL
);

CREATE INDEX idx_workout_exercises_workout_id          ON workout_exercises (workout_id);
CREATE INDEX idx_workout_exercises_exercise_id         ON workout_exercises (exercise_id);
CREATE INDEX idx_workout_exercises_plan_exercise_id    ON workout_exercises (plan_exercise_id);
CREATE INDEX idx_workout_exercises_workout_id_position ON workout_exercises (workout_id, position);

CREATE TABLE workout_sets (
  workout_set_id      UUID        NOT NULL DEFAULT gen_random_uuid(),
  workout_exercise_id UUID        NOT NULL,
  set_number          SMALLINT    CHECK (set_number > 0),
  reps                SMALLINT    CHECK (reps >= 0),
  weight              NUMERIC(6,2) CHECK (weight >= 0),
  rir                 SMALLINT    CHECK (rir >= 0 AND rir <= 10),
  is_warmup           BOOLEAN     NOT NULL DEFAULT FALSE,
  completed_at        TIMESTAMPTZ,
  PRIMARY KEY (workout_set_id),
  UNIQUE (workout_exercise_id, set_number),
  FOREIGN KEY (workout_exercise_id) REFERENCES workout_exercises (workout_exercise_id) ON DELETE CASCADE
);

CREATE INDEX idx_workout_sets_workout_exercise_id ON workout_sets (workout_exercise_id);

-- ============================================================
-- AI Interactions
-- ============================================================

CREATE TABLE ai_interactions (
  ai_interaction_id UUID        NOT NULL DEFAULT gen_random_uuid(),
  user_id           UUID        NOT NULL,
  -- e.g. 'plan' | 'workout' | 'general'
  context_type      VARCHAR(30) CHECK (context_type IN ('plan', 'workout', 'general')),
  -- polymorphic: references plan_id or workout_id
  context_id        UUID        DEFAULT NULL,
  query_text        TEXT,
  response_text     TEXT,
  model_name        VARCHAR(80),
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ai_interaction_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_ai_interactions_user_id          ON ai_interactions (user_id);
CREATE INDEX idx_ai_interactions_context          ON ai_interactions (context_type, context_id);
-- BRIN index for append-only time-series data — much smaller than B-tree
CREATE INDEX idx_ai_interactions_created_at_brin  ON ai_interactions USING BRIN (created_at);

-- ============================================================
-- User Profiles & Medical
-- ============================================================

CREATE TABLE user_profiles (
  user_id        UUID        NOT NULL,
  date_of_birth  DATE,
  sex            VARCHAR(10) CHECK (sex IN ('M', 'F', 'other')),
  -- supports up to 999.9 cm
  height_cm      NUMERIC(5,1) CHECK (height_cm > 0),
  activity_level VARCHAR(30)
    CHECK (activity_level IN ('sedentary', 'lightly_active', 'moderately_active', 'very_active')),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE TABLE user_medical_profiles (
  medical_profile_id UUID        NOT NULL DEFAULT gen_random_uuid(),
  user_id            UUID        NOT NULL,
  has_injuries       BOOLEAN     DEFAULT FALSE,
  injury_details     TEXT,
  surgeries_history  TEXT,
  family_history     TEXT,
  notes              TEXT,
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (medical_profile_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_medical_profiles_user_id ON user_medical_profiles (user_id);

CREATE TABLE user_medications (
  medication_id   UUID         NOT NULL DEFAULT gen_random_uuid(),
  user_id         UUID         NOT NULL,
  medication_name VARCHAR(150) NOT NULL,
  dosage          VARCHAR(60),
  frequency       VARCHAR(60),
  start_date      DATE,
  end_date        DATE,
  notes           TEXT,
  CHECK (end_date IS NULL OR start_date IS NULL OR end_date >= start_date),
  PRIMARY KEY (medication_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_medications_user_id         ON user_medications (user_id);
CREATE INDEX idx_user_medications_medication_name ON user_medications (medication_name);

CREATE TABLE user_allergies (
  allergy_id UUID         NOT NULL DEFAULT gen_random_uuid(),
  user_id    UUID         NOT NULL,
  allergen   VARCHAR(120) NOT NULL,
  reaction   VARCHAR(120),
  severity   VARCHAR(20)  CHECK (severity IN ('mild', 'moderate', 'severe')),
  notes      TEXT,
  PRIMARY KEY (allergy_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_allergies_user_id  ON user_allergies (user_id);
CREATE INDEX idx_user_allergies_allergen ON user_allergies (allergen);

-- ============================================================
-- Calorie Tracking
-- ============================================================

CREATE TABLE calorie_targets (
  calorie_target_id    UUID        NOT NULL DEFAULT gen_random_uuid(),
  user_id              UUID        NOT NULL,
  -- realistic range ~1000–9999 kcal
  maintenance_calories SMALLINT    NOT NULL CHECK (maintenance_calories > 0),
  -- e.g. formula | ai | manual
  method               VARCHAR(20) CHECK (method IN ('formula', 'ai', 'manual')),
  effective_from       DATE        NOT NULL,
  -- NULL = currently active target
  effective_to         DATE        DEFAULT NULL,
  notes                TEXT,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (effective_to IS NULL OR effective_to >= effective_from),
  PRIMARY KEY (calorie_target_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_calorie_targets_user_id                ON calorie_targets (user_id);
CREATE INDEX idx_calorie_targets_user_id_effective_from ON calorie_targets (user_id, effective_from);
-- Partial index: quickly find the single active target per user
CREATE UNIQUE INDEX idx_calorie_targets_active_per_user
  ON calorie_targets (user_id)
  WHERE effective_to IS NULL;

CREATE TABLE calorie_intake_logs (
  calorie_log_id    UUID        NOT NULL DEFAULT gen_random_uuid(),
  user_id           UUID        NOT NULL,
  log_date          DATE        NOT NULL,
  calories_consumed SMALLINT    NOT NULL CHECK (calories_consumed >= 0),
  notes             TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (calorie_log_id),
  UNIQUE (user_id, log_date),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_calorie_intake_logs_user_id ON calorie_intake_logs (user_id);

-- ============================================================
-- Sleep Tracking
-- ============================================================

CREATE TABLE sleep_targets (
  sleep_target_id   UUID         NOT NULL DEFAULT gen_random_uuid(),
  user_id           UUID         NOT NULL,
  -- e.g. 7.5 hours; max 24.0
  target_sleep_hours NUMERIC(4,1) NOT NULL CHECK (target_sleep_hours > 0 AND target_sleep_hours <= 24),
  effective_from    DATE         NOT NULL,
  effective_to      DATE         DEFAULT NULL,
  notes             TEXT,
  created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  CHECK (effective_to IS NULL OR effective_to >= effective_from),
  PRIMARY KEY (sleep_target_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_sleep_targets_user_id                ON sleep_targets (user_id);
CREATE INDEX idx_sleep_targets_user_id_effective_from ON sleep_targets (user_id, effective_from);
-- Partial index: quickly find the single active sleep target per user
CREATE UNIQUE INDEX idx_sleep_targets_active_per_user
  ON sleep_targets (user_id)
  WHERE effective_to IS NULL;

CREATE TABLE sleep_duration_logs (
  sleep_log_id          UUID         NOT NULL DEFAULT gen_random_uuid(),
  user_id               UUID         NOT NULL,
  log_date              DATE         NOT NULL,
  sleep_duration_hours  NUMERIC(4,1) NOT NULL CHECK (sleep_duration_hours > 0 AND sleep_duration_hours <= 24),
  notes                 TEXT,
  created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  PRIMARY KEY (sleep_log_id),
  UNIQUE (user_id, log_date),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_sleep_duration_logs_user_id ON sleep_duration_logs (user_id);

-- ============================================================
-- Weight Tracking
-- ============================================================

CREATE TABLE weight_logs (
  weight_log_id        UUID         NOT NULL DEFAULT gen_random_uuid(),
  user_id              UUID         NOT NULL,
  logged_at            TIMESTAMPTZ  NOT NULL,
  -- e.g. 0–999.99 kg
  weight_kg            NUMERIC(6,2) NOT NULL CHECK (weight_kg > 0),
  -- 0.00–99.99 %
  body_fat_percentage  NUMERIC(5,2) DEFAULT NULL CHECK (body_fat_percentage >= 0 AND body_fat_percentage < 100),
  notes                TEXT,
  PRIMARY KEY (weight_log_id),
  FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX idx_weight_logs_user_id            ON weight_logs (user_id);
CREATE INDEX idx_weight_logs_user_id_logged_at  ON weight_logs (user_id, logged_at DESC);

-- ============================================================
-- Notes
-- ============================================================
--
-- UUID generation:
--   gen_random_uuid() is built-in since PostgreSQL 13.
--   For PG 12 and below, enable pgcrypto: CREATE EXTENSION IF NOT EXISTS pgcrypto;
--   and use gen_random_uuid() from that extension instead.
--
-- Timestamps:
--   All timestamps use TIMESTAMPTZ (timestamp with time zone).
--   Store everything in UTC from your application layer.
--   Convert to user's local timezone at display time, not storage time.
--
-- BRIN indexes:
--   Used on append-only time-series columns (ai_interactions.created_at).
--   BRIN stores min/max per block range — ~100x smaller than B-tree,
--   very fast for range scans on naturally ordered data.
--
-- Partial (filtered) indexes:
--   Used for "active target" lookups on calorie_targets and sleep_targets.
--   Also enforces the business rule that only one active target per user
--   can exist at a time (UNIQUE partial index on user_id WHERE effective_to IS NULL).
--
-- CHECK constraints:
--   Used in place of PostgreSQL ENUM type for low-cardinality string columns.
--   Easier to ALTER (adding a new value to PG ENUM requires a full catalog lock).
-- ============================================================
