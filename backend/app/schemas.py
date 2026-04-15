from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, model_validator


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
    # Required by backend but optional from app (sensible defaults)
    age: int = Field(ge=13, le=100, default=25)
    sex: str = "other"
    height_cm: float = Field(gt=80, lt=260, default=170.0)
    weight_kg: float = Field(gt=25, lt=350, default=70.0)
    activity_level: str = "moderate"

    # App sends goal_type; backend uses goal_name
    goal_name: str = "general_fitness"
    goal_type: str | None = None

    days_per_week: int = Field(ge=2, le=6, default=3)
    experience_level: str = "beginner"

    # App sends equipment as a string; backend expects list[str]
    equipment: list[str] | str = []

    # App sends constraints; backend uses injuries
    injuries: str = ""
    constraints: str | None = None

    conditions: list[str] = []
    medications: list[str] = []
    allergies: list[str] = []
    calorie_target: int | None = None
    sleep_target_hours: float | None = None

    @model_validator(mode="after")
    def _normalize(self):
        # Map goal_type → goal_name
        if self.goal_type:
            self.goal_name = self.goal_type
        # Map constraints → injuries
        if self.constraints and not self.injuries:
            self.injuries = self.constraints
        # Accept equipment as plain string
        if isinstance(self.equipment, str):
            self.equipment = [self.equipment] if self.equipment else []
        return self


class PlanGenerationRequest(BaseModel):
    goal_name: str | None = None
    goal_type: str | None = None  # app alias for goal_name
    days_per_week: int | None = None
    equipment: list[str] | str | None = None
    experience_level: str | None = None
    constraints: str | None = None

    @model_validator(mode="after")
    def _normalize(self):
        if self.goal_type and not self.goal_name:
            self.goal_name = self.goal_type
        if isinstance(self.equipment, str):
            self.equipment = [self.equipment] if self.equipment else []
        return self


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
