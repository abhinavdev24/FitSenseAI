from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    name: str
    email: EmailStr
    needs_onboarding: bool


class OnboardingRequest(BaseModel):
    age: int = Field(ge=13, le=100)
    sex: str
    height_cm: float = Field(gt=80, lt=260)
    weight_kg: float = Field(gt=25, lt=350)
    goal_name: str
    days_per_week: int = Field(ge=2, le=6)
    experience_level: str
    activity_level: str
    equipment: list[str] = []
    injuries: str = ""
    conditions: list[str] = []
    medications: list[str] = []
    allergies: list[str] = []
    calorie_target: int | None = None
    sleep_target_hours: float | None = None


class PlanGenerationRequest(BaseModel):
    goal_name: str | None = None
    days_per_week: int | None = None
    equipment: list[str] | None = None
    experience_level: str | None = None
    constraints: str | None = None


class PlanModifyRequest(BaseModel):
    instruction: str
    constraints: str | None = None


class WorkoutCreateRequest(BaseModel):
    plan_id: str | None = None
    plan_day_id: str | None = None
    started_at: datetime | None = None
    notes: str | None = None


class WorkoutExerciseCreateRequest(BaseModel):
    exercise_id: str
    plan_exercise_id: str | None = None
    position: int = 1
    notes: str | None = None


class WorkoutSetCreateRequest(BaseModel):
    workout_exercise_id: str
    set_number: int
    reps: int
    weight: float = 0
    rir: int = 2
    is_warmup: bool = False
    completed_at: datetime | None = None


class DailySleepRequest(BaseModel):
    logged_on: date
    hours: float = Field(gt=0, lt=24)


class DailyCaloriesRequest(BaseModel):
    logged_on: date
    calories: int = Field(gt=0, lt=10000)


class DailyWeightRequest(BaseModel):
    logged_at: datetime
    weight_kg: float = Field(gt=20, lt=350)
    body_fat_percentage: float | None = None


class CoachRequest(BaseModel):
    message: str
    context_mode: str | None = None


class AdaptationRequest(BaseModel):
    days_window: int = Field(default=14, ge=7, le=60)


class StandardMessage(BaseModel):
    message: str


class DashboardResponse(BaseModel):
    profile: dict[str, Any]
    current_plan: dict[str, Any] | None
    recent_workouts: list[dict[str, Any]]
    recent_logs: dict[str, Any]


class PlanJobEnqueueResponse(BaseModel):
    job_id: str
    status: str
    job_type: str
    message: str
    latest_plan: dict[str, Any] | None = None
    execution_debug: dict[str, Any] | None = None


class PlanJobStatusResponse(BaseModel):
    job_id: str
    status: str
    job_type: str
    progress_message: str | None = None
    error_message: str | None = None
    latest_plan: dict[str, Any] | None = None
    result_plan: dict[str, Any] | None = None
    execution_debug: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
