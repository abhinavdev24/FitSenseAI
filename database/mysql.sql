-- ============================================================
-- FitSense AI — MySQL Schema
-- Generated from database_design.dbml
-- Optimizations:
--   • BINARY(16) UUIDs (50% smaller than CHAR(36), index-friendly)
--   • TINYINT(1) for booleans
--   • SMALLINT / TINYINT for small-range numerics
--   • DECIMAL(5,2) for weight/body fat; DECIMAL(4,1) for sleep hours
--   • ENUM for low-cardinality string columns
--   • MEDIUMTEXT for base64 thumbnail
--   • DATETIME(0) — no sub-second precision needed
--   • InnoDB + utf8mb4 throughout
--   • UUID ↔ BINARY(16) usage guidelines at bottom
-- ============================================================

SET FOREIGN_KEY_CHECKS = 0;

-- ============================================================
-- Users & Auth
-- ============================================================

CREATE TABLE `users` (
  `user_id`    BINARY(16)   NOT NULL,
  `name`       VARCHAR(120) NOT NULL,
  `email`      VARCHAR(254) NOT NULL,
  `created_at` DATETIME(0)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uq_users_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Goals
-- ============================================================

CREATE TABLE `goals` (
  `goal_id`     BINARY(16)   NOT NULL,
  `name`        VARCHAR(120) NOT NULL,
  `description` TEXT,
  PRIMARY KEY (`goal_id`),
  UNIQUE KEY `uq_goals_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `user_goals` (
  `user_id`  BINARY(16)  NOT NULL,
  `goal_id`  BINARY(16)  NOT NULL,
  `priority` TINYINT UNSIGNED,
  PRIMARY KEY (`user_id`, `goal_id`),
  KEY `idx_user_goals_goal_id` (`goal_id`),
  CONSTRAINT `fk_user_goals_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_user_goals_goal` FOREIGN KEY (`goal_id`) REFERENCES `goals` (`goal_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Conditions
-- ============================================================

CREATE TABLE `conditions` (
  `condition_id` BINARY(16)   NOT NULL,
  `name`         VARCHAR(120) NOT NULL,
  `description`  TEXT,
  PRIMARY KEY (`condition_id`),
  UNIQUE KEY `uq_conditions_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `user_conditions` (
  `user_id`      BINARY(16)  NOT NULL,
  `condition_id` BINARY(16)  NOT NULL,
  -- e.g. mild | moderate | severe
  `severity`     ENUM('mild','moderate','severe') DEFAULT NULL,
  `notes`        TEXT,
  PRIMARY KEY (`user_id`, `condition_id`),
  KEY `idx_user_conditions_condition_id` (`condition_id`),
  CONSTRAINT `fk_user_conditions_user`      FOREIGN KEY (`user_id`)      REFERENCES `users`      (`user_id`)      ON DELETE CASCADE,
  CONSTRAINT `fk_user_conditions_condition` FOREIGN KEY (`condition_id`) REFERENCES `conditions` (`condition_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Equipment & Exercises
-- ============================================================

CREATE TABLE `equipment` (
  `equipment_id` BINARY(16)   NOT NULL,
  `name`         VARCHAR(120) NOT NULL,
  `category`     VARCHAR(60),
  PRIMARY KEY (`equipment_id`),
  UNIQUE KEY `uq_equipment_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `exercises` (
  `exercise_id`        BINARY(16)   NOT NULL,
  `name`               VARCHAR(150) NOT NULL,
  `primary_muscle`     VARCHAR(80),
  `notes`              TEXT,
  -- URL to hosted video file (e.g. S3/CDN)
  `video_url`          VARCHAR(500),
  -- base64-encoded thumbnail; MEDIUMTEXT supports up to ~16 MB
  `thumbnail_base64`   MEDIUMTEXT,
  PRIMARY KEY (`exercise_id`),
  UNIQUE KEY `uq_exercises_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `exercise_equipment` (
  `exercise_id`  BINARY(16) NOT NULL,
  `equipment_id` BINARY(16) NOT NULL,
  PRIMARY KEY (`exercise_id`, `equipment_id`),
  KEY `idx_exercise_equipment_equipment_id` (`equipment_id`),
  CONSTRAINT `fk_exercise_equipment_exercise`  FOREIGN KEY (`exercise_id`)  REFERENCES `exercises`  (`exercise_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_exercise_equipment_equipment` FOREIGN KEY (`equipment_id`) REFERENCES `equipment`  (`equipment_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Workout Plans (Blueprint)
-- ============================================================

CREATE TABLE `workout_plans` (
  `plan_id`    BINARY(16)   NOT NULL,
  `user_id`    BINARY(16)   NOT NULL,
  `name`       VARCHAR(150),
  `is_active`  TINYINT(1)   NOT NULL DEFAULT 0,
  `created_at` DATETIME(0)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`plan_id`),
  KEY `idx_workout_plans_user_id`           (`user_id`),
  KEY `idx_workout_plans_user_id_is_active` (`user_id`, `is_active`),
  CONSTRAINT `fk_workout_plans_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Named days within a plan cycle (e.g. PUSH_1, PULL_2, UPPER_1)
CREATE TABLE `plan_days` (
  `plan_day_id` BINARY(16)   NOT NULL,
  `plan_id`     BINARY(16)   NOT NULL,
  `name`        VARCHAR(60)  NOT NULL,
  -- ordering within the plan cycle
  `day_order`   TINYINT UNSIGNED,
  `notes`       TEXT,
  PRIMARY KEY (`plan_day_id`),
  KEY         `idx_plan_days_plan_id`            (`plan_id`),
  UNIQUE KEY  `uq_plan_days_plan_id_name`        (`plan_id`, `name`),
  KEY         `idx_plan_days_plan_id_day_order`  (`plan_id`, `day_order`),
  CONSTRAINT `fk_plan_days_plan` FOREIGN KEY (`plan_id`) REFERENCES `workout_plans` (`plan_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `plan_exercises` (
  `plan_exercise_id` BINARY(16)  NOT NULL,
  -- references named day, not the plan directly
  `plan_day_id`      BINARY(16)  NOT NULL,
  `exercise_id`      BINARY(16)  NOT NULL,
  `position`         SMALLINT UNSIGNED,
  `notes`            TEXT,
  PRIMARY KEY (`plan_exercise_id`),
  KEY `idx_plan_exercises_plan_day_id`          (`plan_day_id`),
  KEY `idx_plan_exercises_exercise_id`          (`exercise_id`),
  KEY `idx_plan_exercises_plan_day_id_position` (`plan_day_id`, `position`),
  CONSTRAINT `fk_plan_exercises_plan_day`  FOREIGN KEY (`plan_day_id`)  REFERENCES `plan_days`  (`plan_day_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_plan_exercises_exercise`  FOREIGN KEY (`exercise_id`)  REFERENCES `exercises`  (`exercise_id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `plan_sets` (
  `plan_set_id`      BINARY(16)     NOT NULL,
  `plan_exercise_id` BINARY(16)     NOT NULL,
  `set_number`       TINYINT UNSIGNED NOT NULL,
  `target_reps`      TINYINT UNSIGNED,
  -- e.g. 0–500 kg in 0.25 kg increments
  `target_weight`    DECIMAL(6,2),
  -- reps-in-reserve: typically 0–5
  `target_rir`       TINYINT UNSIGNED,
  `rest_seconds`     SMALLINT UNSIGNED,
  PRIMARY KEY (`plan_set_id`),
  KEY         `idx_plan_sets_plan_exercise_id`            (`plan_exercise_id`),
  UNIQUE KEY  `uq_plan_sets_plan_exercise_id_set_number`  (`plan_exercise_id`, `set_number`),
  CONSTRAINT `chk_plan_sets_set_number_positive` CHECK (`set_number` > 0),
  CONSTRAINT `fk_plan_sets_plan_exercise` FOREIGN KEY (`plan_exercise_id`) REFERENCES `plan_exercises` (`plan_exercise_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Workouts (Actual sessions)
-- ============================================================

CREATE TABLE `workouts` (
  `workout_id`  BINARY(16)  NOT NULL,
  `user_id`     BINARY(16)  NOT NULL,
  -- nullable: freeform workout may not follow a plan
  `plan_id`     BINARY(16)  DEFAULT NULL,
  -- nullable: which named day template was followed (e.g. "PUSH_1")
  `plan_day_id` BINARY(16)  DEFAULT NULL,
  `started_at`  DATETIME(0),
  `ended_at`    DATETIME(0),
  `notes`       TEXT,
  PRIMARY KEY (`workout_id`),
  KEY `idx_workouts_user_id`            (`user_id`),
  KEY `idx_workouts_plan_id`            (`plan_id`),
  KEY `idx_workouts_plan_day_id`        (`plan_day_id`),
  KEY `idx_workouts_user_id_started_at` (`user_id`, `started_at`),
  CONSTRAINT `fk_workouts_user`     FOREIGN KEY (`user_id`)     REFERENCES `users`         (`user_id`)     ON DELETE CASCADE,
  CONSTRAINT `fk_workouts_plan`     FOREIGN KEY (`plan_id`)     REFERENCES `workout_plans` (`plan_id`)     ON DELETE SET NULL,
  CONSTRAINT `fk_workouts_plan_day` FOREIGN KEY (`plan_day_id`) REFERENCES `plan_days`     (`plan_day_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `workout_exercises` (
  `workout_exercise_id` BINARY(16)  NOT NULL,
  `workout_id`          BINARY(16)  NOT NULL,
  `exercise_id`         BINARY(16)  NOT NULL,
  -- nullable link to the blueprint row
  `plan_exercise_id`    BINARY(16)  DEFAULT NULL,
  `position`            SMALLINT UNSIGNED,
  `notes`               TEXT,
  PRIMARY KEY (`workout_exercise_id`),
  KEY `idx_workout_exercises_workout_id`           (`workout_id`),
  KEY `idx_workout_exercises_exercise_id`          (`exercise_id`),
  KEY `idx_workout_exercises_plan_exercise_id`     (`plan_exercise_id`),
  KEY `idx_workout_exercises_workout_id_position`  (`workout_id`, `position`),
  CONSTRAINT `fk_workout_exercises_workout`       FOREIGN KEY (`workout_id`)       REFERENCES `workouts`        (`workout_id`)       ON DELETE CASCADE,
  CONSTRAINT `fk_workout_exercises_exercise`      FOREIGN KEY (`exercise_id`)      REFERENCES `exercises`       (`exercise_id`)      ON DELETE RESTRICT,
  CONSTRAINT `fk_workout_exercises_plan_exercise` FOREIGN KEY (`plan_exercise_id`) REFERENCES `plan_exercises`  (`plan_exercise_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `workout_sets` (
  `workout_set_id`      BINARY(16)    NOT NULL,
  `workout_exercise_id` BINARY(16)    NOT NULL,
  `set_number`          TINYINT UNSIGNED NOT NULL,
  `reps`                TINYINT UNSIGNED,
  `weight`              DECIMAL(6,2),
  `rir`                 TINYINT UNSIGNED,
  `is_warmup`           TINYINT(1)    NOT NULL DEFAULT 0,
  `completed_at`        DATETIME(0),
  PRIMARY KEY (`workout_set_id`),
  KEY         `idx_workout_sets_workout_exercise_id`              (`workout_exercise_id`),
  UNIQUE KEY  `uq_workout_sets_workout_exercise_id_set_number`    (`workout_exercise_id`, `set_number`),
  CONSTRAINT `fk_workout_sets_workout_exercise` FOREIGN KEY (`workout_exercise_id`) REFERENCES `workout_exercises` (`workout_exercise_id`) ON DELETE CASCADE,
  CONSTRAINT `chk_workout_sets_set_number_positive` CHECK (`set_number` > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- AI Interactions
-- ============================================================

CREATE TABLE `ai_interactions` (
  `ai_interaction_id` BINARY(16)  NOT NULL,
  `user_id`           BINARY(16)  NOT NULL,
  -- e.g. 'plan' | 'workout' | 'general'
  `context_type`      VARCHAR(30) DEFAULT NULL,
  -- polymorphic: references plan_id or workout_id
  `context_id`        BINARY(16)  DEFAULT NULL,
  `query_text`        TEXT,
  `response_text`     TEXT,
  `model_name`        VARCHAR(80),
  `created_at`        DATETIME(0) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`ai_interaction_id`),
  KEY `idx_ai_interactions_user_id`              (`user_id`),
  KEY `idx_ai_interactions_context_type_id`      (`context_type`, `context_id`),
  KEY `idx_ai_interactions_created_at`           (`created_at`),
  CONSTRAINT `fk_ai_interactions_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- User Profiles & Medical
-- ============================================================

CREATE TABLE `user_profiles` (
  `user_id`        BINARY(16)  NOT NULL,
  `date_of_birth`  DATE,
  -- M | F | other
  `sex`            ENUM('M','F','other') DEFAULT NULL,
  -- supports up to 999.9 cm
  `height_cm`      DECIMAL(5,1),
  `activity_level` ENUM('sedentary','lightly_active','moderately_active','very_active') DEFAULT NULL,
  `updated_at`     DATETIME(0) NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`user_id`),
  CONSTRAINT `fk_user_profiles_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `user_medical_profiles` (
  `medical_profile_id` BINARY(16)  NOT NULL,
  `user_id`            BINARY(16)  NOT NULL,
  `has_injuries`       TINYINT(1)  DEFAULT 0,
  `injury_details`     TEXT,
  `surgeries_history`  TEXT,
  `family_history`     TEXT,
  `notes`              TEXT,
  `updated_at`         DATETIME(0) NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`medical_profile_id`),
  KEY `idx_user_medical_profiles_user_id` (`user_id`),
  CONSTRAINT `fk_user_medical_profiles_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `user_medications` (
  `medication_id`   BINARY(16)   NOT NULL,
  `user_id`         BINARY(16)   NOT NULL,
  `medication_name` VARCHAR(150) NOT NULL,
  `dosage`          VARCHAR(60),
  `frequency`       VARCHAR(60),
  `start_date`      DATE,
  `end_date`        DATE,
  `notes`           TEXT,
  PRIMARY KEY (`medication_id`),
  KEY `idx_user_medications_user_id`         (`user_id`),
  KEY `idx_user_medications_medication_name` (`medication_name`),
  CONSTRAINT `fk_user_medications_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `user_allergies` (
  `allergy_id` BINARY(16)   NOT NULL,
  `user_id`    BINARY(16)   NOT NULL,
  `allergen`   VARCHAR(120) NOT NULL,
  `reaction`   VARCHAR(120),
  `severity`   ENUM('mild','moderate','severe') DEFAULT NULL,
  `notes`      TEXT,
  PRIMARY KEY (`allergy_id`),
  KEY `idx_user_allergies_user_id`  (`user_id`),
  KEY `idx_user_allergies_allergen` (`allergen`),
  CONSTRAINT `fk_user_allergies_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Calorie Tracking
-- ============================================================

CREATE TABLE `calorie_targets` (
  `calorie_target_id`   BINARY(16)  NOT NULL,
  `user_id`             BINARY(16)  NOT NULL,
  `maintenance_calories` SMALLINT UNSIGNED NOT NULL,  -- realistic range ~1000–9999
  -- e.g. formula | ai | manual
  `method`              ENUM('formula','ai','manual') DEFAULT NULL,
  `effective_from`      DATE        NOT NULL,
  -- nullable = currently active target
  `effective_to`        DATE        DEFAULT NULL,
  `notes`               TEXT,
  `created_at`          DATETIME(0) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`calorie_target_id`),
  KEY `idx_calorie_targets_user_id`                (`user_id`),
  KEY `idx_calorie_targets_user_id_effective_from` (`user_id`, `effective_from`),
  CONSTRAINT `fk_calorie_targets_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `calorie_intake_logs` (
  `calorie_log_id`   BINARY(16)       NOT NULL,
  `user_id`          BINARY(16)       NOT NULL,
  `log_date`         DATE             NOT NULL,
  `calories_consumed` SMALLINT UNSIGNED NOT NULL,
  `notes`            TEXT,
  `created_at`       DATETIME(0)      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`calorie_log_id`),
  KEY        `idx_calorie_intake_logs_user_id`          (`user_id`),
  UNIQUE KEY `uq_calorie_intake_logs_user_id_log_date`  (`user_id`, `log_date`),
  CONSTRAINT `fk_calorie_intake_logs_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Sleep Tracking
-- ============================================================

CREATE TABLE `sleep_targets` (
  `sleep_target_id`   BINARY(16)    NOT NULL,
  `user_id`           BINARY(16)    NOT NULL,
  -- e.g. 7.5 hours; max 24.0
  `target_sleep_hours` DECIMAL(4,1) NOT NULL,
  `effective_from`    DATE          NOT NULL,
  `effective_to`      DATE          DEFAULT NULL,
  `notes`             TEXT,
  `created_at`        DATETIME(0)   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`sleep_target_id`),
  KEY `idx_sleep_targets_user_id`                (`user_id`),
  KEY `idx_sleep_targets_user_id_effective_from` (`user_id`, `effective_from`),
  CONSTRAINT `fk_sleep_targets_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `sleep_duration_logs` (
  `sleep_log_id`        BINARY(16)    NOT NULL,
  `user_id`             BINARY(16)    NOT NULL,
  `log_date`            DATE          NOT NULL,
  `sleep_duration_hours` DECIMAL(4,1) NOT NULL,
  `notes`               TEXT,
  `created_at`          DATETIME(0)   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`sleep_log_id`),
  KEY        `idx_sleep_duration_logs_user_id`          (`user_id`),
  UNIQUE KEY `uq_sleep_duration_logs_user_id_log_date`  (`user_id`, `log_date`),
  CONSTRAINT `fk_sleep_duration_logs_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Weight Tracking
-- ============================================================

CREATE TABLE `weight_logs` (
  `weight_log_id`      BINARY(16)   NOT NULL,
  `user_id`            BINARY(16)   NOT NULL,
  `logged_at`          DATETIME(0)  NOT NULL,
  -- e.g. 999.99 kg
  `weight_kg`          DECIMAL(6,2) NOT NULL,
  -- e.g. 0.00–99.99 %
  `body_fat_percentage` DECIMAL(5,2) DEFAULT NULL,
  `notes`              TEXT,
  PRIMARY KEY (`weight_log_id`),
  KEY `idx_weight_logs_user_id`            (`user_id`),
  KEY `idx_weight_logs_user_id_logged_at`  (`user_id`, `logged_at`),
  CONSTRAINT `fk_weight_logs_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SET FOREIGN_KEY_CHECKS = 1;

-- ============================================================
-- UUID Usage Guidelines (MySQL 8.0+ built-in functions)
-- How to convert between application UUIDs and BINARY(16)
-- ============================================================
--
-- Store:    INSERT ... VALUES (UUID_TO_BIN(?, TRUE), ...)
-- Retrieve: SELECT BIN_TO_UUID(user_id, TRUE) AS user_id ...
--
-- The second argument `TRUE` swaps the time-high and time-low fields
-- of UUIDv1, improving index locality. For UUIDv4/v7 it is a no-op.
--
-- UUID_TO_BIN() and BIN_TO_UUID() are built into MySQL 8.0+;
-- no custom helper functions are required.
-- ============================================================
