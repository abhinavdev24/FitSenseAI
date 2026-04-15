from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    profile: Mapped["UserProfile"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    medical_profile: Mapped["UserMedicalProfile"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    preference: Mapped["UserPreference"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    goals: Mapped[list["UserGoal"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    conditions: Mapped[list["UserCondition"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    medications: Mapped[list["UserMedication"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    allergies: Mapped[list["UserAllergy"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    plans: Mapped[list["WorkoutPlan"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    workouts: Mapped[list["Workout"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["SessionToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    ai_interactions: Mapped[list["AIInteraction"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    plan_jobs: Mapped[list["PlanGenerationJob"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    calorie_targets: Mapped[list["CalorieTarget"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sleep_targets: Mapped[list["SleepTarget"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class SessionToken(Base):
    __tablename__ = "session_tokens"

    token: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="sessions")


class Goal(Base):
    __tablename__ = "goals"

    goal_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class UserGoal(Base):
    __tablename__ = "user_goals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.goal_id"), index=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped[User] = relationship(back_populates="goals")
    goal: Mapped[Goal] = relationship()


class Condition(Base):
    __tablename__ = "conditions"

    condition_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class UserCondition(Base):
    __tablename__ = "user_conditions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    condition_id: Mapped[str] = mapped_column(ForeignKey("conditions.condition_id"), index=True)
    severity: Mapped[str | None] = mapped_column(String(20), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(back_populates="conditions")
    condition: Mapped[Condition] = relationship()


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), primary_key=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(10), nullable=True)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    activity_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="profile")


class UserMedicalProfile(Base):
    __tablename__ = "user_medical_profiles"

    medical_profile_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), unique=True, index=True)
    has_injuries: Mapped[bool] = mapped_column(Boolean, default=False)
    injury_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    surgeries_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    family_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="medical_profile")


class UserMedication(Base):
    __tablename__ = "user_medications"

    medication_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    medication_name: Mapped[str] = mapped_column(String(150))
    dosage: Mapped[str | None] = mapped_column(String(60), nullable=True)
    frequency: Mapped[str | None] = mapped_column(String(60), nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(back_populates="medications")


class UserAllergy(Base):
    __tablename__ = "user_allergies"

    allergy_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    allergen: Mapped[str] = mapped_column(String(120))
    reaction: Mapped[str | None] = mapped_column(String(120), nullable=True)
    severity: Mapped[str | None] = mapped_column(String(20), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(back_populates="allergies")


class UserPreference(Base):
    __tablename__ = "user_preferences"

    preference_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), unique=True, index=True)
    days_per_week: Mapped[int] = mapped_column(Integer, default=4)
    experience_level: Mapped[str] = mapped_column(String(30), default="beginner")
    equipment_csv: Mapped[str] = mapped_column(Text, default="bodyweight,dumbbells")

    user: Mapped[User] = relationship(back_populates="preference")


class Equipment(Base):
    __tablename__ = "equipment"

    equipment_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    category: Mapped[str | None] = mapped_column(String(60), nullable=True)

    exercise_links: Mapped[list["ExerciseEquipment"]] = relationship(cascade="all, delete-orphan")


class Exercise(Base):
    __tablename__ = "exercises"

    exercise_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(150), unique=True)
    primary_muscle: Mapped[str | None] = mapped_column(String(80), nullable=True)
    category: Mapped[str] = mapped_column(String(60), default="general")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    thumbnail_base64: Mapped[str | None] = mapped_column(Text, nullable=True)
    equipment_csv: Mapped[str] = mapped_column(Text, default="bodyweight")

    equipment_links: Mapped[list["ExerciseEquipment"]] = relationship(cascade="all, delete-orphan")


class WorkoutPlan(Base):
    __tablename__ = "workout_plans"

    plan_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    name: Mapped[str] = mapped_column(String(150), default="Personalized Plan")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="plans")
    days: Mapped[list["PlanDay"]] = relationship(back_populates="plan", cascade="all, delete-orphan")
    workouts: Mapped[list["Workout"]] = relationship(back_populates="plan")


class PlanDay(Base):
    __tablename__ = "plan_days"

    plan_day_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(ForeignKey("workout_plans.plan_id"), index=True)
    name: Mapped[str] = mapped_column(String(60))
    day_order: Mapped[int] = mapped_column(Integer, default=1)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    plan: Mapped[WorkoutPlan] = relationship(back_populates="days")
    exercises: Mapped[list["PlanExercise"]] = relationship(back_populates="day", cascade="all, delete-orphan")


class PlanExercise(Base):
    __tablename__ = "plan_exercises"

    plan_exercise_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    plan_day_id: Mapped[str] = mapped_column(ForeignKey("plan_days.plan_day_id"), index=True)
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.exercise_id"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=1)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    day: Mapped[PlanDay] = relationship(back_populates="exercises")
    exercise: Mapped[Exercise] = relationship()
    sets: Mapped[list["PlanSet"]] = relationship(back_populates="plan_exercise", cascade="all, delete-orphan")


class PlanSet(Base):
    __tablename__ = "plan_sets"

    plan_set_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    plan_exercise_id: Mapped[str] = mapped_column(ForeignKey("plan_exercises.plan_exercise_id"), index=True)
    set_number: Mapped[int] = mapped_column(Integer)
    target_reps: Mapped[int] = mapped_column(Integer)
    target_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_rir: Mapped[int] = mapped_column(Integer)
    rest_seconds: Mapped[int] = mapped_column(Integer)

    plan_exercise: Mapped[PlanExercise] = relationship(back_populates="sets")


class Workout(Base):
    __tablename__ = "workouts"

    workout_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    plan_id: Mapped[str | None] = mapped_column(ForeignKey("workout_plans.plan_id"), nullable=True)
    plan_day_id: Mapped[str | None] = mapped_column(ForeignKey("plan_days.plan_day_id"), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(back_populates="workouts")
    plan: Mapped[WorkoutPlan | None] = relationship(back_populates="workouts")
    exercises: Mapped[list["WorkoutExercise"]] = relationship(back_populates="workout", cascade="all, delete-orphan")


class WorkoutExercise(Base):
    __tablename__ = "workout_exercises"

    workout_exercise_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workout_id: Mapped[str] = mapped_column(ForeignKey("workouts.workout_id"), index=True)
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.exercise_id"), index=True)
    plan_exercise_id: Mapped[str | None] = mapped_column(ForeignKey("plan_exercises.plan_exercise_id"), nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=1)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    workout: Mapped[Workout] = relationship(back_populates="exercises")
    exercise: Mapped[Exercise] = relationship()
    sets: Mapped[list["WorkoutSet"]] = relationship(back_populates="workout_exercise", cascade="all, delete-orphan")


class WorkoutSet(Base):
    __tablename__ = "workout_sets"

    workout_set_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workout_exercise_id: Mapped[str] = mapped_column(ForeignKey("workout_exercises.workout_exercise_id"), index=True)
    set_number: Mapped[int] = mapped_column(Integer)
    reps: Mapped[int] = mapped_column(Integer)
    weight: Mapped[float] = mapped_column(Float, default=0)
    rir: Mapped[int] = mapped_column(Integer, default=2)
    is_warmup: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    workout_exercise: Mapped[WorkoutExercise] = relationship(back_populates="sets")


class CalorieIntakeLog(Base):
    __tablename__ = "calorie_intake_logs"

    calorie_log_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    logged_on: Mapped[date] = mapped_column(Date)
    calories: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SleepDurationLog(Base):
    __tablename__ = "sleep_duration_logs"

    sleep_log_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    logged_on: Mapped[date] = mapped_column(Date)
    hours: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WeightLog(Base):
    __tablename__ = "weight_logs"

    weight_log_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    logged_at: Mapped[datetime] = mapped_column(DateTime)
    weight_kg: Mapped[float] = mapped_column(Float)
    body_fat_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)


class ExerciseEquipment(Base):
    __tablename__ = "exercise_equipment"

    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.exercise_id"), primary_key=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.equipment_id"), primary_key=True)

    exercise: Mapped["Exercise"] = relationship(back_populates="equipment_links")
    equipment: Mapped["Equipment"] = relationship(back_populates="exercise_links")


class CalorieTarget(Base):
    __tablename__ = "calorie_targets"

    calorie_target_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    maintenance_calories: Mapped[int] = mapped_column(Integer)
    method: Mapped[str | None] = mapped_column(String(20), nullable=True)
    effective_from: Mapped[date] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="calorie_targets")


class SleepTarget(Base):
    __tablename__ = "sleep_targets"

    sleep_target_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    target_sleep_hours: Mapped[float] = mapped_column(Float)
    effective_from: Mapped[date] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="sleep_targets")


class PlanGenerationJob(Base):
    __tablename__ = "plan_generation_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    job_type: Mapped[str] = mapped_column(String(30), default="generate")
    status: Mapped[str] = mapped_column(String(30), default="queued")
    request_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_plan_id: Mapped[str | None] = mapped_column(ForeignKey("workout_plans.plan_id"), nullable=True)
    progress_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="plan_jobs")
    result_plan: Mapped[WorkoutPlan | None] = relationship()


class AIInteraction(Base):
    __tablename__ = "ai_interactions"

    ai_interaction_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    context_type: Mapped[str] = mapped_column(String(30), default="general")
    context_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="ai_interactions")
