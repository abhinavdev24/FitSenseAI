from __future__ import annotations

import json
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.local")
from datetime import datetime
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .database import Base, SessionLocal, engine, get_db
from . import models, schemas, services
from .llm_runtime import get_runtime

app = FastAPI(title="FitSenseAI Backend", version="1.1.0")

bearer_scheme = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        services.seed_reference_data(db)
    finally:
        db.close()


def _extract_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials.strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
        )
    return credentials.credentials.strip()


def get_current_user(
    db: Annotated[Session, Depends(get_db)],
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> models.User:
    token = _extract_token(credentials)
    session = db.get(models.SessionToken, token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    user = db.get(models.User, session.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session"
        )
    return user


def _hydrate_user(db: Session, user_id: str) -> models.User:
    return (
        db.execute(
            select(models.User)
            .where(models.User.user_id == user_id)
            .options(
                joinedload(models.User.goals).joinedload(models.UserGoal.goal),
                joinedload(models.User.conditions).joinedload(
                    models.UserCondition.condition
                ),
                joinedload(models.User.profile),
                joinedload(models.User.preference),
                joinedload(models.User.medical_profile),
                joinedload(models.User.plans),
            )
        )
        .unique()
        .scalars()
        .one()
    )


@app.get("/")
def root() -> dict:
    return {
        "name": "FitSenseAI Backend",
        "status": "ok",
        "docs": "/docs",
        "architecture": "async-plan-jobs-enabled",
    }


@app.post("/auth/signup", response_model=schemas.AuthResponse)
def signup(payload: schemas.SignupRequest, db: Annotated[Session, Depends(get_db)]):
    existing = db.scalar(select(models.User).where(models.User.email == payload.email))
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = models.User(
        name=payload.name,
        email=payload.email,
        password_hash=services.hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = models.SessionToken(user_id=user.user_id)
    db.add(token)
    db.commit()
    return schemas.AuthResponse(
        token=token.token,
        user_id=user.user_id,
        name=user.name,
        email=user.email,
        needs_onboarding=True,
    )


@app.post("/auth/login", response_model=schemas.AuthResponse)
def login(payload: schemas.LoginRequest, db: Annotated[Session, Depends(get_db)]):
    user = db.scalar(select(models.User).where(models.User.email == payload.email))
    if user is None or not services.verify_password(
        payload.password, user.password_hash
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = models.SessionToken(user_id=user.user_id)
    db.add(token)
    db.commit()
    needs_onboarding = (
        user.profile is None or user.preference is None or len(user.goals) == 0
    )
    return schemas.AuthResponse(
        token=token.token,
        user_id=user.user_id,
        name=user.name,
        email=user.email,
        needs_onboarding=needs_onboarding,
    )


@app.get("/me")
def me(
    user: Annotated[models.User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    hydrated = _hydrate_user(db, user.user_id)
    return services.build_profile_summary(hydrated)


@app.get("/model/runtime")
def model_runtime(user: Annotated[models.User, Depends(get_current_user)]):
    return services._student_runtime_info()


@app.post("/profile/onboarding")
def save_onboarding(
    payload: schemas.OnboardingRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    services.upsert_onboarding(db, user, payload)
    hydrated = _hydrate_user(db, user.user_id)
    return {
        "message": "Profile saved",
        "profile": services.build_profile_summary(hydrated),
    }


@app.get("/catalog/exercises")
def list_exercises(db: Annotated[Session, Depends(get_db)]):
    exercises = db.scalars(
        select(models.Exercise).order_by(models.Exercise.name.asc())
    ).all()
    return [
        {
            "exercise_id": ex.exercise_id,
            "name": ex.name,
            "primary_muscle": ex.primary_muscle,
            "category": ex.category,
            "equipment": ex.equipment_csv.split(","),
        }
        for ex in exercises
    ]


@app.post("/plans", response_model=schemas.PlanJobEnqueueResponse)
def create_plan(
    payload: schemas.PlanGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    hydrated = _hydrate_user(db, user.user_id)
    job = services.enqueue_plan_job(db, hydrated, payload, job_type="generate")
    background_tasks.add_task(services.process_plan_job, SessionLocal, job.job_id)
    latest_plan = services.get_current_plan(db, hydrated.user_id)
    return schemas.PlanJobEnqueueResponse(
        job_id=job.job_id,
        status=job.status,
        job_type=job.job_type,
        message="Plan generation queued. The app can keep showing the latest plan while the pipeline runs.",
        latest_plan=services.serialize_plan(latest_plan) if latest_plan else None,
        execution_debug=services._student_runtime_info(),
    )


@app.post("/pipeline/trigger", response_model=schemas.PlanJobEnqueueResponse)
def trigger_pipeline(
    payload: schemas.PlanGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    return create_plan(payload, background_tasks, db, user)


@app.get("/plans/jobs/{job_id}", response_model=schemas.PlanJobStatusResponse)
def get_plan_job_status(
    job_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    job = services.get_plan_job(db, job_id, user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Plan job not found")
    return services.serialize_job(db, job)


@app.get("/plans/jobs/latest")
def get_latest_plan_job(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    job = services.get_latest_pending_plan_job(db, user.user_id)
    if job is None:
        return {"job": None}
    return {"job": services.serialize_job(db, job)}


@app.get("/plans/current")
def get_current_plan_endpoint(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    plan = services.get_current_plan(db, user.user_id)
    active_job = services.get_latest_pending_plan_job(db, user.user_id)
    return {
        "plan": services.serialize_plan(plan) if plan else None,
        "active_job": services.serialize_job(db, active_job) if active_job else None,
    }


@app.post("/plans/{plan_id}:modify", response_model=schemas.PlanJobEnqueueResponse)
def modify_plan_endpoint(
    plan_id: str,
    payload: schemas.PlanModifyRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    active = services.get_current_plan(db, user.user_id)
    if active is None or active.plan_id != plan_id:
        raise HTTPException(status_code=404, detail="Active plan not found")
    request_like = schemas.PlanGenerationRequest(constraints=payload.constraints)
    job = services.enqueue_plan_job(
        db, user, request_like, job_type="modify", instruction=payload.instruction
    )
    background_tasks.add_task(services.process_plan_job, SessionLocal, job.job_id)
    return schemas.PlanJobEnqueueResponse(
        job_id=job.job_id,
        status=job.status,
        job_type=job.job_type,
        message="Plan modification queued. The current plan stays visible until the updated one is ready.",
        latest_plan=services.serialize_plan(active),
        execution_debug=services._student_runtime_info(),
    )


@app.post("/workouts")
def create_workout(
    payload: schemas.WorkoutCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    workout = models.Workout(
        user_id=user.user_id,
        plan_id=payload.plan_id,
        plan_day_id=payload.plan_day_id,
        started_at=payload.started_at or datetime.utcnow(),
        notes=payload.notes,
    )
    db.add(workout)
    db.commit()
    return {"workout_id": workout.workout_id}


@app.post("/workouts/{workout_id}/exercises")
def create_workout_exercise(
    workout_id: str,
    payload: schemas.WorkoutExerciseCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    workout = db.get(models.Workout, workout_id)
    if workout is None or workout.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Workout not found")
    row = models.WorkoutExercise(
        workout_id=workout_id,
        exercise_id=payload.exercise_id,
        plan_exercise_id=payload.plan_exercise_id,
        position=payload.position,
        notes=payload.notes,
    )
    db.add(row)
    db.commit()
    return {"workout_exercise_id": row.workout_exercise_id}


@app.post("/workouts/{workout_id}/sets")
def create_workout_set(
    workout_id: str,
    payload: schemas.WorkoutSetCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    workout = db.get(models.Workout, workout_id)
    if workout is None or workout.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Workout not found")
    ex = db.get(models.WorkoutExercise, payload.workout_exercise_id)
    if ex is None or ex.workout_id != workout_id:
        raise HTTPException(status_code=404, detail="Workout exercise not found")
    row = models.WorkoutSet(
        workout_exercise_id=payload.workout_exercise_id,
        set_number=payload.set_number,
        reps=payload.reps,
        weight=payload.weight,
        rir=payload.rir,
        is_warmup=payload.is_warmup,
        completed_at=payload.completed_at or datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    return {"workout_set_id": row.workout_set_id}


@app.get("/workouts/recent")
def get_recent_workouts(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    return {"workouts": services.recent_workouts_summary(db, user.user_id)}


@app.post("/daily/sleep")
def create_sleep_log(
    payload: schemas.DailySleepRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    row = models.SleepDurationLog(
        user_id=user.user_id, logged_on=payload.logged_on, hours=payload.hours
    )
    db.add(row)
    db.commit()
    return {"sleep_log_id": row.sleep_log_id}


@app.post("/daily/calories")
def create_calorie_log(
    payload: schemas.DailyCaloriesRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    row = models.CalorieIntakeLog(
        user_id=user.user_id, logged_on=payload.logged_on, calories=payload.calories
    )
    db.add(row)
    db.commit()
    return {"calorie_log_id": row.calorie_log_id}


@app.post("/daily/weight")
def create_weight_log(
    payload: schemas.DailyWeightRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    row = models.WeightLog(
        user_id=user.user_id,
        logged_at=payload.logged_at,
        weight_kg=payload.weight_kg,
        body_fat_percentage=payload.body_fat_percentage,
    )
    db.add(row)
    db.commit()
    return {"weight_log_id": row.weight_log_id}


@app.get("/dashboard", response_model=schemas.DashboardResponse)
def dashboard(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    hydrated = _hydrate_user(db, user.user_id)
    plan = services.get_current_plan(db, user.user_id)
    active_job = services.get_latest_pending_plan_job(db, user.user_id)
    return schemas.DashboardResponse(
        profile={
            **services.build_profile_summary(hydrated),
            "active_plan_job": (
                services.serialize_job(db, active_job) if active_job else None
            ),
        },
        current_plan=services.serialize_plan(plan) if plan else None,
        recent_workouts=services.recent_workouts_summary(db, user.user_id),
        recent_logs=services.recent_logs_summary(db, user.user_id),
    )


@app.post("/targets/calories")
def create_calorie_target(
    payload: schemas.CalorieTargetRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    row = models.CalorieTarget(
        user_id=user.user_id,
        maintenance_calories=payload.maintenance_calories,
        method=payload.method,
        effective_from=payload.effective_from,
        effective_to=payload.effective_to,
        notes=payload.notes,
    )
    db.add(row)
    db.commit()
    return {"calorie_target_id": row.calorie_target_id}


@app.get("/targets/calories")
def get_calorie_targets(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    targets = db.scalars(
        select(models.CalorieTarget)
        .where(models.CalorieTarget.user_id == user.user_id)
        .order_by(models.CalorieTarget.effective_from.desc())
    ).all()
    return [
        {
            "calorie_target_id": t.calorie_target_id,
            "maintenance_calories": t.maintenance_calories,
            "method": t.method,
            "effective_from": str(t.effective_from),
            "effective_to": str(t.effective_to) if t.effective_to else None,
            "notes": t.notes,
        }
        for t in targets
    ]


@app.post("/targets/sleep")
def create_sleep_target(
    payload: schemas.SleepTargetRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    row = models.SleepTarget(
        user_id=user.user_id,
        target_sleep_hours=payload.target_sleep_hours,
        effective_from=payload.effective_from,
        effective_to=payload.effective_to,
        notes=payload.notes,
    )
    db.add(row)
    db.commit()
    return {"sleep_target_id": row.sleep_target_id}


@app.get("/targets/sleep")
def get_sleep_targets(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    targets = db.scalars(
        select(models.SleepTarget)
        .where(models.SleepTarget.user_id == user.user_id)
        .order_by(models.SleepTarget.effective_from.desc())
    ).all()
    return [
        {
            "sleep_target_id": t.sleep_target_id,
            "target_sleep_hours": t.target_sleep_hours,
            "effective_from": str(t.effective_from),
            "effective_to": str(t.effective_to) if t.effective_to else None,
            "notes": t.notes,
        }
        for t in targets
    ]


@app.post("/coach")
def coach(
    payload: schemas.CoachRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    hydrated = (
        db.execute(
            select(models.User)
            .where(models.User.user_id == user.user_id)
            .options(
                joinedload(models.User.goals).joinedload(models.UserGoal.goal),
                joinedload(models.User.medical_profile),
            )
        )
        .unique()
        .scalars()
        .one()
    )
    recent = services.recent_workouts_summary(db, user.user_id)
    logs = services.recent_logs_summary(db, user.user_id)
    reply, context_type, execution_debug = services.build_coach_reply(
        hydrated, payload.message, recent, logs
    )
    runtime_info = get_runtime().info()
    interaction = models.AIInteraction(
        user_id=user.user_id,
        context_type="coach",
        query_text=execution_debug.get("llm_query") if isinstance(execution_debug, dict) and execution_debug.get("llm_query") else payload.message,
        response_text=reply,
        model_name=(
            runtime_info.base_model
            if context_type == "student-llm"
            else "local-rule-engine-v1"
        ),
    )
    db.add(interaction)
    db.commit()
    return {
        "reply": reply,
        "context_type": context_type,
        "execution_debug": execution_debug,
    }


@app.get("/coach/stream")
def coach_stream(
    message: str = Query(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    hydrated = (
        db.execute(
            select(models.User)
            .where(models.User.user_id == user.user_id)
            .options(
                joinedload(models.User.goals).joinedload(models.UserGoal.goal),
                joinedload(models.User.medical_profile),
            )
        )
        .unique()
        .scalars()
        .one()
    )
    recent = services.recent_workouts_summary(db, user.user_id)
    logs = services.recent_logs_summary(db, user.user_id)
    reply, context_type, execution_debug = services.build_coach_reply(
        hydrated, message, recent, logs
    )
    runtime_info = get_runtime().info()
    interaction = models.AIInteraction(
        user_id=user.user_id,
        context_type="coach-stream",
        query_text=execution_debug.get("llm_query") if isinstance(execution_debug, dict) and execution_debug.get("llm_query") else message,
        response_text=reply,
        model_name=(
            runtime_info.base_model
            if context_type == "student-llm"
            else "local-rule-engine-v1"
        ),
    )
    db.add(interaction)
    db.commit()

    def event_gen():
        yield f"data: {json.dumps({'debug': execution_debug})}\n\n"
        for token in reply.split():
            yield f"data: {json.dumps({'delta': token + ' '})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/adaptation:next_week")
def adaptation(
    payload: schemas.AdaptationRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[models.User, Depends(get_current_user)],
):
    hydrated = (
        db.execute(
            select(models.User)
            .where(models.User.user_id == user.user_id)
            .options(joinedload(models.User.preference))
        )
        .unique()
        .scalars()
        .one()
    )
    return services.compute_adaptation(db, hydrated, payload.days_window)
