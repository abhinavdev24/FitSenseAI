from __future__ import annotations

import json
import math
import re
import time
from datetime import date, datetime, timedelta
from typing import Iterable

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from . import models
from .llm_runtime import get_runtime

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


GOAL_SEED = {
    "fat loss": "Reduce body fat while keeping training sustainable.",
    "muscle gain": "Build muscle with steady progression and enough recovery.",
    "strength": "Improve performance on the main movement patterns.",
    "general fitness": "Build consistency, movement quality, and broad fitness.",
}

CONDITION_SEED = {
    "knee pain": "Lower-body exercise selection should stay joint-friendly.",
    "lower back pain": "Prioritize spine-friendly hinging and bracing.",
    "shoulder irritation": "Avoid painful overhead loading and deep ranges.",
    "hypertension": "Keep effort submaximal and breathing controlled.",
}

EXERCISE_SEED = [
    {
        "name": "Goblet Squat",
        "primary_muscle": "quads",
        "category": "lower",
        "equipment_csv": "dumbbells",
    },
    {
        "name": "Leg Press",
        "primary_muscle": "quads",
        "category": "lower",
        "equipment_csv": "machines",
    },
    {
        "name": "Romanian Deadlift",
        "primary_muscle": "hamstrings",
        "category": "lower",
        "equipment_csv": "barbell,dumbbells",
    },
    {
        "name": "Hip Thrust",
        "primary_muscle": "glutes",
        "category": "lower",
        "equipment_csv": "barbell,bodyweight",
    },
    {
        "name": "Bench Press",
        "primary_muscle": "chest",
        "category": "push",
        "equipment_csv": "barbell",
    },
    {
        "name": "Push-Up",
        "primary_muscle": "chest",
        "category": "push",
        "equipment_csv": "bodyweight",
    },
    {
        "name": "Incline Dumbbell Press",
        "primary_muscle": "chest",
        "category": "push",
        "equipment_csv": "dumbbells",
    },
    {
        "name": "Seated Dumbbell Shoulder Press",
        "primary_muscle": "shoulders",
        "category": "push",
        "equipment_csv": "dumbbells",
    },
    {
        "name": "Lateral Raise",
        "primary_muscle": "shoulders",
        "category": "push",
        "equipment_csv": "dumbbells",
    },
    {
        "name": "Cable Triceps Pushdown",
        "primary_muscle": "triceps",
        "category": "push",
        "equipment_csv": "cables,machines",
    },
    {
        "name": "Lat Pulldown",
        "primary_muscle": "lats",
        "category": "pull",
        "equipment_csv": "cables,machines",
    },
    {
        "name": "Chest-Supported Row",
        "primary_muscle": "upper back",
        "category": "pull",
        "equipment_csv": "dumbbells,machines",
    },
    {
        "name": "Seated Cable Row",
        "primary_muscle": "upper back",
        "category": "pull",
        "equipment_csv": "cables",
    },
    {
        "name": "Dumbbell Curl",
        "primary_muscle": "biceps",
        "category": "pull",
        "equipment_csv": "dumbbells",
    },
    {
        "name": "Plank",
        "primary_muscle": "core",
        "category": "core",
        "equipment_csv": "bodyweight",
    },
    {
        "name": "Walking Lunge",
        "primary_muscle": "quads",
        "category": "lower",
        "equipment_csv": "bodyweight,dumbbells",
    },
    {
        "name": "Step-Up",
        "primary_muscle": "quads",
        "category": "lower",
        "equipment_csv": "bodyweight,dumbbells",
    },
    {
        "name": "Glute Bridge",
        "primary_muscle": "glutes",
        "category": "lower",
        "equipment_csv": "bodyweight",
    },
    {
        "name": "Landmine Press",
        "primary_muscle": "shoulders",
        "category": "push",
        "equipment_csv": "barbell",
    },
    {
        "name": "Face Pull",
        "primary_muscle": "rear delts",
        "category": "pull",
        "equipment_csv": "cables",
    },
]

EQUIPMENT_SEED = [
    ("bodyweight", "minimal"),
    ("dumbbells", "free_weights"),
    ("barbell", "free_weights"),
    ("machines", "machines"),
    ("cables", "machines"),
]


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def seed_reference_data(db: Session) -> None:
    if db.scalar(select(models.Goal).limit(1)) is None:
        for name, description in GOAL_SEED.items():
            db.add(models.Goal(name=name, description=description))
    if db.scalar(select(models.Condition).limit(1)) is None:
        for name, description in CONDITION_SEED.items():
            db.add(models.Condition(name=name, description=description))
    if db.scalar(select(models.Equipment).limit(1)) is None:
        for name, category in EQUIPMENT_SEED:
            db.add(models.Equipment(name=name, category=category))
    if db.scalar(select(models.Exercise).limit(1)) is None:
        for item in EXERCISE_SEED:
            db.add(models.Exercise(**item))
    db.commit()


AVAILABLE_SPLITS = {
    2: ["Full Body A", "Full Body B"],
    3: ["Upper", "Lower", "Full Body"],
    4: ["Upper 1", "Lower 1", "Upper 2", "Lower 2"],
    5: ["Push", "Pull", "Legs", "Upper", "Lower"],
    6: ["Push 1", "Pull 1", "Legs 1", "Push 2", "Pull 2", "Legs 2"],
}


DAY_CATEGORIES = {
    "Upper": ["push", "pull", "core"],
    "Upper 1": ["push", "pull", "core"],
    "Upper 2": ["push", "pull", "core"],
    "Lower": ["lower", "core"],
    "Lower 1": ["lower", "core"],
    "Lower 2": ["lower", "core"],
    "Push": ["push", "core"],
    "Push 1": ["push", "core"],
    "Push 2": ["push", "core"],
    "Pull": ["pull", "core"],
    "Pull 1": ["pull", "core"],
    "Pull 2": ["pull", "core"],
    "Legs": ["lower", "core"],
    "Legs 1": ["lower", "core"],
    "Legs 2": ["lower", "core"],
    "Full Body": ["push", "pull", "lower", "core"],
    "Full Body A": ["push", "pull", "lower", "core"],
    "Full Body B": ["push", "pull", "lower", "core"],
}


def _get_or_create_goal(db: Session, goal_name: str) -> models.Goal:
    goal_name = goal_name.strip().lower()
    goal = db.scalar(select(models.Goal).where(models.Goal.name == goal_name))
    if goal:
        return goal
    goal = models.Goal(
        name=goal_name, description=GOAL_SEED.get(goal_name, "User-selected goal.")
    )
    db.add(goal)
    db.commit()
    db.refresh(goal)
    return goal


def _get_or_create_condition(db: Session, name: str) -> models.Condition:
    n = name.strip().lower()
    condition = db.scalar(select(models.Condition).where(models.Condition.name == n))
    if condition:
        return condition
    condition = models.Condition(
        name=n, description=CONDITION_SEED.get(n, "User-entered condition.")
    )
    db.add(condition)
    db.commit()
    db.refresh(condition)
    return condition


def upsert_onboarding(db: Session, user: models.User, payload) -> None:
    dob = date.today() - timedelta(days=payload.age * 365)
    if user.profile is None:
        user.profile = models.UserProfile(user_id=user.user_id)
    user.profile.date_of_birth = dob
    user.profile.sex = payload.sex
    user.profile.height_cm = payload.height_cm
    user.profile.activity_level = payload.activity_level
    user.profile.updated_at = datetime.utcnow()

    if user.preference is None:
        user.preference = models.UserPreference(user_id=user.user_id)
    user.preference.days_per_week = payload.days_per_week
    user.preference.experience_level = payload.experience_level
    user.preference.equipment_csv = ",".join(
        sorted(set(payload.equipment or ["bodyweight", "dumbbells"]))
    )

    if user.medical_profile is None:
        user.medical_profile = models.UserMedicalProfile(user_id=user.user_id)
    user.medical_profile.has_injuries = bool(payload.injuries.strip())
    user.medical_profile.injury_details = payload.injuries.strip() or None
    user.medical_profile.updated_at = datetime.utcnow()

    user.goals.clear()
    goal = _get_or_create_goal(db, payload.goal_name)
    user.goals.append(models.UserGoal(goal_id=goal.goal_id, priority=0))

    user.conditions.clear()
    for cond in payload.conditions:
        c = _get_or_create_condition(db, cond)
        user.conditions.append(
            models.UserCondition(condition_id=c.condition_id, severity="moderate")
        )

    user.medications.clear()
    for med in payload.medications:
        if med.strip():
            user.medications.append(models.UserMedication(medication_name=med.strip()))

    user.allergies.clear()
    for allergy in payload.allergies:
        if allergy.strip():
            user.allergies.append(models.UserAllergy(allergen=allergy.strip()))

    db.add(
        models.WeightLog(
            user_id=user.user_id,
            logged_at=datetime.utcnow(),
            weight_kg=payload.weight_kg,
        )
    )
    if payload.calorie_target:
        db.add(
            models.CalorieIntakeLog(
                user_id=user.user_id,
                logged_on=date.today(),
                calories=payload.calorie_target,
            )
        )
    if payload.sleep_target_hours:
        db.add(
            models.SleepDurationLog(
                user_id=user.user_id,
                logged_on=date.today(),
                hours=payload.sleep_target_hours,
            )
        )
    db.commit()


def _available_equipment(
    user: models.User, override: list[str] | None = None
) -> set[str]:
    if override:
        return {e.strip().lower() for e in override if e.strip()}
    if user.preference and user.preference.equipment_csv:
        return {
            e.strip().lower()
            for e in user.preference.equipment_csv.split(",")
            if e.strip()
        }
    return {"bodyweight", "dumbbells"}


def _pick_exercises(
    db: Session, categories: Iterable[str], equipment: set[str], injuries: str
) -> list[models.Exercise]:
    all_exercises = db.scalars(select(models.Exercise)).all()
    picks = []
    injury_text = (injuries or "").lower()
    for category in categories:
        candidates = [e for e in all_exercises if e.category == category]
        filtered = []
        for ex in candidates:
            ex_eq = {s.strip().lower() for s in ex.equipment_csv.split(",")}
            if not ex_eq.intersection(equipment) and "bodyweight" not in ex_eq:
                continue
            name = ex.name.lower()
            if (
                "knee" in injury_text
                and ("lunge" in name or "squat" in name)
                and "goblet" not in name
            ):
                continue
            if "shoulder" in injury_text and ("overhead" in name or "bench" in name):
                continue
            if "back" in injury_text and "deadlift" in name:
                continue
            filtered.append(ex)
        picks.extend(filtered[:2] if category != "core" else filtered[:1])
    seen = set()
    result = []
    for ex in picks:
        if ex.exercise_id not in seen:
            seen.add(ex.exercise_id)
            result.append(ex)
    return result[:6]


def _training_targets(goal_name: str, experience: str) -> tuple[int, int, int, int]:
    goal_name = (goal_name or "general fitness").lower()
    experience = (experience or "beginner").lower()
    base_sets = 3 if experience == "beginner" else 4
    if "strength" in goal_name:
        return base_sets, 5, 2, 150
    if "muscle" in goal_name:
        return base_sets, 10, 2, 90
    if "fat" in goal_name:
        return base_sets, 12, 3, 75
    return base_sets, 10, 3, 90


def serialize_plan(plan: models.WorkoutPlan) -> dict:
    return {
        "plan_id": plan.plan_id,
        "name": plan.name,
        "is_active": plan.is_active,
        "explanation": plan.explanation,
        "created_at": plan.created_at.isoformat(),
        "days": [
            {
                "plan_day_id": day.plan_day_id,
                "name": day.name,
                "day_order": day.day_order,
                "exercises": [
                    {
                        "plan_exercise_id": pe.plan_exercise_id,
                        "exercise_id": pe.exercise.exercise_id,
                        "exercise_name": pe.exercise.name,
                        "position": pe.position,
                        "notes": pe.notes,
                        "sets": [
                            {
                                "plan_set_id": ps.plan_set_id,
                                "set_number": ps.set_number,
                                "target_reps": ps.target_reps,
                                "target_rir": ps.target_rir,
                                "rest_seconds": ps.rest_seconds,
                                "target_weight": ps.target_weight,
                            }
                            for ps in sorted(pe.sets, key=lambda x: x.set_number)
                        ],
                    }
                    for pe in sorted(day.exercises, key=lambda x: x.position)
                ],
            }
            for day in sorted(plan.days, key=lambda x: x.day_order)
        ],
    }


def _runtime_debug_payload(
    *,
    endpoint: str,
    selected_backend: str,
    fallback_reason: str | None = None,
    runtime_info=None,
    notes: str | None = None,
) -> dict:
    info = runtime_info or get_runtime().info()
    return {
        "endpoint": endpoint,
        "selected_backend": selected_backend,
        "fallback_reason": fallback_reason,
        "notes": notes,
        "runtime": (
            info.to_dict()
            if hasattr(info, "to_dict")
            else {
                "available": getattr(info, "available", None),
                "provider": getattr(info, "provider", None),
                "base_model": getattr(info, "base_model", None),
                "adapter_path": getattr(info, "adapter_path", None),
                "registry_record": getattr(info, "registry_record", None),
                "reason": getattr(info, "reason", None),
            }
        ),
    }


def _student_user_message(
    user: models.User,
    request,
    instruction: str | None = None,
    active_plan: models.WorkoutPlan | None = None,
    recent_workouts: list[dict] | None = None,
    logs: dict | None = None,
) -> str:
    profile = build_profile_summary(user)
    payload = {
        "goal_name": getattr(request, "goal_name", None),
        "days_per_week": getattr(request, "days_per_week", None),
        "equipment": getattr(request, "equipment", None),
        "experience_level": getattr(request, "experience_level", None),
        "constraints": getattr(request, "constraints", None),
        "instruction": instruction,
        "profile": profile,
    }
    if active_plan is not None:
        payload["active_plan"] = serialize_plan(active_plan)
    if recent_workouts:
        payload["recent_workouts"] = recent_workouts
    if logs:
        payload["logs"] = logs
    return json.dumps(payload, ensure_ascii=False)


def _ensure_exercise(db: Session, exercise_name: str) -> models.Exercise:
    name = (exercise_name or "").strip() or "Bodyweight Movement"
    ex = db.scalar(select(models.Exercise).where(models.Exercise.name == name))
    if ex:
        return ex
    inferred_category = (
        "core" if "plank" in name.lower() or "dead bug" in name.lower() else "general"
    )
    ex = models.Exercise(
        name=name,
        primary_muscle=None,
        category=inferred_category,
        equipment_csv="bodyweight",
    )
    db.add(ex)
    db.flush()
    return ex


def _normalize_llm_plan_json(plan_json: dict) -> dict:
    """Normalize slightly different student-model JSON shapes into the backend schema."""
    normalized = {
        "plan_name": plan_json.get("plan_name")
        or plan_json.get("name")
        or "Student LLM Plan",
        "explanation": plan_json.get("explanation"),
        "days": [],
    }

    days = plan_json.get("days", []) or []
    for day_idx, day_payload in enumerate(days, start=1):
        day_payload = day_payload or {}
        norm_day = {
            "name": day_payload.get("name") or f"Day {day_idx}",
            "day_order": day_payload.get("day_order") or day_idx,
            "notes": day_payload.get("notes"),
            "exercises": [],
        }

        exercises = day_payload.get("exercises", []) or []
        for ex_idx, ex_payload in enumerate(exercises, start=1):
            ex_payload = ex_payload or {}
            raw_position = ex_payload.get("position")
            raw_notes = ex_payload.get("notes")

            norm_ex = {
                "exercise_name": ex_payload.get("exercise_name")
                or ex_payload.get("name")
                or "Bodyweight Movement",
                "position": raw_position if isinstance(raw_position, int) else ex_idx,
                "notes": raw_notes if isinstance(raw_notes, str) else None,
                "sets": [],
            }

            sets_payload = ex_payload.get("sets", []) or []

            if isinstance(sets_payload, int):
                reps = int(ex_payload.get("reps") or 10)
                rir = int(ex_payload.get("target_rir") or 3)
                rest = int(ex_payload.get("rest_seconds") or 60)
                sets_payload = [
                    {
                        "set_number": sidx,
                        "target_reps": reps,
                        "target_rir": rir,
                        "rest_seconds": rest,
                    }
                    for sidx in range(1, max(1, sets_payload) + 1)
                ]
            elif not sets_payload and any(
                k in ex_payload for k in ("reps", "rest_seconds", "target_reps")
            ):
                sets_payload = [
                    {
                        "set_number": 1,
                        "target_reps": int(
                            ex_payload.get("target_reps")
                            or ex_payload.get("reps")
                            or 10
                        ),
                        "target_rir": int(ex_payload.get("target_rir") or 3),
                        "rest_seconds": int(ex_payload.get("rest_seconds") or 60),
                    }
                ]

            if isinstance(raw_position, str):
                extra = f"Position cue: {raw_position}"
                norm_ex["notes"] = (
                    f"{norm_ex['notes']}; {extra}" if norm_ex["notes"] else extra
                )

            for sidx, set_payload in enumerate(sets_payload or [], start=1):
                set_payload = set_payload or {}
                norm_ex["sets"].append(
                    {
                        "set_number": int(set_payload.get("set_number") or sidx),
                        "target_reps": int(
                            set_payload.get("target_reps")
                            or set_payload.get("reps")
                            or 10
                        ),
                        "target_rir": int(set_payload.get("target_rir") or 3),
                        "rest_seconds": int(set_payload.get("rest_seconds") or 60),
                        "target_weight": set_payload.get("target_weight"),
                    }
                )

            if not norm_ex["sets"]:
                norm_ex["sets"] = [
                    {
                        "set_number": 1,
                        "target_reps": 10,
                        "target_rir": 3,
                        "rest_seconds": 60,
                        "target_weight": None,
                    }
                ]

            norm_day["exercises"].append(norm_ex)

        normalized["days"].append(norm_day)

    return normalized


def _persist_llm_plan(
    db: Session, user: models.User, plan_json: dict, *, explanation: str | None = None
) -> models.WorkoutPlan:
    plan_json = _normalize_llm_plan_json(plan_json or {})
    for plan in user.plans:
        plan.is_active = False

    plan = models.WorkoutPlan(
        user_id=user.user_id,
        name=plan_json.get("plan_name") or "Student LLM Plan",
        is_active=True,
        explanation=explanation or plan_json.get("explanation"),
    )
    db.add(plan)
    db.flush()

    for idx, day_payload in enumerate(plan_json.get("days", []) or [], start=1):
        day = models.PlanDay(
            plan_id=plan.plan_id,
            name=(day_payload or {}).get("name") or f"Day {idx}",
            day_order=(day_payload or {}).get("day_order") or idx,
            notes=(day_payload or {}).get("notes"),
        )
        db.add(day)
        db.flush()

        for pos, ex_payload in enumerate(
            (day_payload or {}).get("exercises", []) or [], start=1
        ):
            ex_model = _ensure_exercise(
                db,
                (ex_payload or {}).get("exercise_name") or "Bodyweight Movement",
            )
            pe = models.PlanExercise(
                plan_day_id=day.plan_day_id,
                exercise_id=ex_model.exercise_id,
                position=(ex_payload or {}).get("position") or pos,
                notes=(ex_payload or {}).get("notes"),
            )
            db.add(pe)
            db.flush()

            sets_payload = (ex_payload or {}).get("sets", []) or []
            if not sets_payload:
                sets_payload = [
                    {
                        "set_number": 1,
                        "target_reps": 10,
                        "target_rir": 3,
                        "rest_seconds": 60,
                    }
                ]

            for sidx, set_payload in enumerate(sets_payload, start=1):
                db.add(
                    models.PlanSet(
                        plan_exercise_id=pe.plan_exercise_id,
                        set_number=(set_payload or {}).get("set_number") or sidx,
                        target_reps=int((set_payload or {}).get("target_reps") or 10),
                        target_rir=int((set_payload or {}).get("target_rir") or 3),
                        rest_seconds=int((set_payload or {}).get("rest_seconds") or 60),
                        target_weight=(set_payload or {}).get("target_weight"),
                    )
                )

    db.commit()

    refreshed = (
        db.execute(
            select(models.WorkoutPlan)
            .where(models.WorkoutPlan.plan_id == plan.plan_id)
            .options(
                joinedload(models.WorkoutPlan.days)
                .joinedload(models.PlanDay.exercises)
                .joinedload(models.PlanExercise.sets),
                joinedload(models.WorkoutPlan.days)
                .joinedload(models.PlanDay.exercises)
                .joinedload(models.PlanExercise.exercise),
            )
        )
        .unique()
        .scalars()
        .one()
    )

    return refreshed


def try_student_plan_generation(
    db: Session,
    user: models.User,
    request,
    instruction: str | None = None,
    active_plan: models.WorkoutPlan | None = None,
):
    runtime = get_runtime()
    runtime_info = runtime.info()
    print(
        f"[student-plan] available={runtime_info.available} adapter={runtime_info.adapter_path} reason={runtime_info.reason} detail={getattr(runtime_info, 'detail', None)}"
    )

    if not runtime_info.available and not runtime._can_use_cloud():
        fallback_reason = runtime_info.reason or "student runtime unavailable"
        return (
            None,
            None,
            _runtime_debug_payload(
                endpoint="plan",
                selected_backend="rules",
                fallback_reason=fallback_reason,
                runtime_info=runtime_info,
                notes=getattr(runtime_info, "detail", None),
            ),
        )

    recent_workouts = recent_workouts_summary(db, user.user_id, limit=5)
    logs = recent_logs_summary(db, user.user_id)
    user_message = _student_user_message(
        user,
        request,
        instruction=instruction,
        active_plan=active_plan,
        recent_workouts=recent_workouts,
        logs=logs,
    )
    plan_json = runtime.generate_plan_json(
        user_message=user_message,
        is_modification=bool(instruction),
    )

    if not plan_json:
        refreshed = runtime.info()
        fallback_reason = (
            refreshed.last_load_error
            or refreshed.reason
            or "student runtime returned no valid JSON"
        )
        print(
            f"[student-plan] generation returned no valid JSON; falling back reason={fallback_reason}"
        )
        return (
            None,
            None,
            _runtime_debug_payload(
                endpoint="plan",
                selected_backend="rules",
                fallback_reason=fallback_reason,
                runtime_info=refreshed,
                notes="Student model was attempted first, but its output could not be used.",
            ),
        )

    explanation = plan_json.get("explanation") or (
        "Modified plan produced by student LLM."
        if instruction
        else "Generated by student LLM."
    )

    plan = _persist_llm_plan(db, user, plan_json, explanation=explanation)
    return (
        plan,
        explanation,
        _runtime_debug_payload(
            endpoint="plan",
            selected_backend="student_model",
            runtime_info=runtime.info(),
            notes="Student adapter was available and produced a valid response.",
        ),
    )


def try_student_coach_reply(
    user: models.User, message: str, recent_workouts: list[dict], logs: dict, current_plan: dict | None = None
):
    runtime = get_runtime()
    runtime_info = runtime.info()
    print(
        f"[student-coach] available={runtime_info.available} adapter={runtime_info.adapter_path} reason={runtime_info.reason} detail={getattr(runtime_info, 'detail', None)}"
    )

    if not runtime_info.available and not runtime._can_use_cloud():
        return None, _runtime_debug_payload(
            endpoint="coach",
            selected_backend="rules",
            fallback_reason=runtime_info.reason or "student runtime unavailable",
            runtime_info=runtime_info,
            notes=getattr(runtime_info, "detail", None),
        )

    payload = {
        "profile": build_profile_summary(user),
        "message": message,
        "recent_workouts": recent_workouts,
        "logs": logs,
    }
    if current_plan is not None:
        payload["current_plan"] = current_plan
    reply = runtime.generate_coach_text(
        user_message=json.dumps(payload, ensure_ascii=False)
    )

    if not reply:
        refreshed = runtime.info()
        fallback_reason = (
            refreshed.last_load_error
            or refreshed.reason
            or "student runtime returned no text"
        )
        print(
            f"[student-coach] generation returned no text; falling back reason={fallback_reason}"
        )
        return None, _runtime_debug_payload(
            endpoint="coach",
            selected_backend="rules",
            fallback_reason=fallback_reason,
            runtime_info=refreshed,
            notes="Student model was attempted first, but its output could not be used.",
        )

    return reply, _runtime_debug_payload(
        endpoint="coach",
        selected_backend="student_model",
        runtime_info=runtime.info(),
        notes="Student adapter was available and produced a valid reply.",
    )


def generate_plan(
    db: Session, user: models.User, request
) -> tuple[models.WorkoutPlan, str, dict]:
    llm_plan, llm_explanation, debug = try_student_plan_generation(db, user, request)
    if llm_plan is not None:
        return (
            llm_plan,
            llm_explanation or llm_plan.explanation or "Generated by student LLM.",
            debug,
        )
    print(
        f"[fallback] using rule-based generate_plan reason={debug.get('fallback_reason') if debug else None}"
    )
    days_per_week = request.days_per_week or (
        user.preference.days_per_week if user.preference else 4
    )
    goal_name = request.goal_name or (
        user.goals[0].goal.name if user.goals else "general fitness"
    )
    experience = request.experience_level or (
        user.preference.experience_level if user.preference else "beginner"
    )
    equipment = _available_equipment(user, request.equipment)
    injuries = (
        user.medical_profile.injury_details if user.medical_profile else ""
    ) or ""
    split = AVAILABLE_SPLITS.get(days_per_week, AVAILABLE_SPLITS[4])

    for plan in user.plans:
        plan.is_active = False

    plan = models.WorkoutPlan(
        user_id=user.user_id,
        name=f"{goal_name.title()} {days_per_week}-Day Plan",
        is_active=True,
    )
    db.add(plan)
    db.flush()

    sets_count, reps, rir, rest = _training_targets(goal_name, experience)
    total_exercises = 0
    for idx, day_name in enumerate(split, start=1):
        day = models.PlanDay(plan_id=plan.plan_id, name=day_name, day_order=idx)
        db.add(day)
        db.flush()
        exercises = _pick_exercises(db, DAY_CATEGORIES[day_name], equipment, injuries)
        for pos, ex in enumerate(exercises, start=1):
            pe = models.PlanExercise(
                plan_day_id=day.plan_day_id, exercise_id=ex.exercise_id, position=pos
            )
            db.add(pe)
            db.flush()
            total_exercises += 1
            for set_no in range(1, sets_count + 1):
                db.add(
                    models.PlanSet(
                        plan_exercise_id=pe.plan_exercise_id,
                        set_number=set_no,
                        target_reps=max(5, reps - (1 if set_no == sets_count else 0)),
                        target_rir=min(4, rir + (1 if "core" in ex.category else 0)),
                        rest_seconds=rest,
                        target_weight=None,
                    )
                )

    explanation = (
        f"Built a {days_per_week}-day plan for {goal_name.lower()} with {total_exercises} exercise slots. "
        f"The structure matches your equipment ({', '.join(sorted(equipment))}) and stays conservative around "
        f"the constraint profile we have on file. Target effort stays around RIR {rir} so you can progress without grinding every set."
    )
    plan.explanation = explanation
    db.commit()
    plan = (
        db.execute(
            select(models.WorkoutPlan)
            .where(models.WorkoutPlan.plan_id == plan.plan_id)
            .options(
                joinedload(models.WorkoutPlan.days)
                .joinedload(models.PlanDay.exercises)
                .joinedload(models.PlanExercise.sets),
                joinedload(models.WorkoutPlan.days)
                .joinedload(models.PlanDay.exercises)
                .joinedload(models.PlanExercise.exercise),
            )
        )
        .unique()
        .scalars()
        .one()
    )
    debug = debug or _runtime_debug_payload(
        endpoint="plan",
        selected_backend="rules",
        fallback_reason="Unknown fallback reason.",
    )
    return plan, explanation, debug


def get_current_plan(db: Session, user_id: str) -> models.WorkoutPlan | None:
    return (
        db.execute(
            select(models.WorkoutPlan)
            .where(
                models.WorkoutPlan.user_id == user_id,
                models.WorkoutPlan.is_active.is_(True),
            )
            .options(
                joinedload(models.WorkoutPlan.days)
                .joinedload(models.PlanDay.exercises)
                .joinedload(models.PlanExercise.sets),
                joinedload(models.WorkoutPlan.days)
                .joinedload(models.PlanDay.exercises)
                .joinedload(models.PlanExercise.exercise),
            )
        )
        .unique()
        .scalars()
        .first()
    )


def modify_plan(
    db: Session, user: models.User, active_plan: models.WorkoutPlan, instruction: str
) -> tuple[models.WorkoutPlan, str, dict]:
    request_like_for_llm = type(
        "Req",
        (),
        {
            "goal_name": None,
            "days_per_week": None,
            "equipment": None,
            "experience_level": None,
            "constraints": instruction,
        },
    )()
    llm_plan, llm_explanation, debug = try_student_plan_generation(
        db, user, request_like_for_llm, instruction=instruction, active_plan=active_plan
    )
    if llm_plan is not None:
        explanation = (
            f"Updated your active plan using the student LLM with the request: '{instruction}'. "
            f"{llm_explanation or ''}"
        ).strip()
        llm_plan.explanation = explanation
        db.commit()
        refreshed = get_current_plan(db, user.user_id)
        return refreshed, explanation, debug
    print(
        f"[fallback] using rule-based modify_plan reason={debug.get('fallback_reason') if debug else None}"
    )
    instruction_lower = instruction.lower()
    schedule_match = re.search(r"(\d)\s*-?day", instruction_lower)
    override_days = int(schedule_match.group(1)) if schedule_match else None
    request_like = type(
        "Req",
        (),
        {
            "goal_name": None,
            "days_per_week": override_days,
            "equipment": None,
            "experience_level": None,
            "constraints": instruction,
        },
    )()
    new_plan, _, fallback_debug = generate_plan(db, user, request_like)

    if any(
        k in instruction_lower
        for k in ["easy", "recover", "dial back", "poor recovery"]
    ):
        for day in new_plan.days:
            for pe in day.exercises:
                pe.sets = pe.sets[:-1] if len(pe.sets) > 2 else pe.sets
                for s in pe.sets:
                    s.target_rir = min(5, s.target_rir + 1)
                    s.rest_seconds += 15
    elif any(k in instruction_lower for k in ["volume", "plateau", "progress", "more"]):
        for day in new_plan.days:
            for pe in day.exercises[:2]:
                last = max((s.set_number for s in pe.sets), default=0)
                pe.sets.append(
                    models.PlanSet(
                        plan_exercise_id=pe.plan_exercise_id,
                        set_number=last + 1,
                        target_reps=pe.sets[-1].target_reps,
                        target_rir=max(1, pe.sets[-1].target_rir - 1),
                        rest_seconds=pe.sets[-1].rest_seconds,
                        target_weight=None,
                    )
                )
    if any(k in instruction_lower for k in ["injury", "pain", "swap"]):
        for day in new_plan.days:
            for pe in day.exercises:
                if pe.exercise.name in {
                    "Bench Press",
                    "Walking Lunge",
                    "Romanian Deadlift",
                    "Seated Dumbbell Shoulder Press",
                }:
                    replacement_name = {
                        "Bench Press": "Incline Dumbbell Press",
                        "Walking Lunge": "Step-Up",
                        "Romanian Deadlift": "Glute Bridge",
                        "Seated Dumbbell Shoulder Press": "Landmine Press",
                    }[pe.exercise.name]
                    replacement = db.scalar(
                        select(models.Exercise).where(
                            models.Exercise.name == replacement_name
                        )
                    )
                    if replacement:
                        pe.exercise_id = replacement.exercise_id
                        pe.exercise = replacement
    explanation = (
        f"Updated your active plan using the request: '{instruction}'. "
        f"I kept the weekly structure aligned to your profile and used conservative substitutions or volume changes where appropriate."
    )
    new_plan.explanation = explanation
    db.commit()
    refreshed = get_current_plan(db, user.user_id)
    return refreshed, explanation, fallback_debug


def build_profile_summary(user: models.User) -> dict:
    return {
        "user_id": user.user_id,
        "name": user.name,
        "email": user.email,
        "goal": user.goals[0].goal.name if user.goals else None,
        "days_per_week": user.preference.days_per_week if user.preference else 4,
        "experience_level": (
            user.preference.experience_level if user.preference else "beginner"
        ),
        "equipment": (
            user.preference.equipment_csv.split(",")
            if user.preference and user.preference.equipment_csv
            else []
        ),
        "injuries": (
            user.medical_profile.injury_details if user.medical_profile else None
        ),
        "conditions": [uc.condition.name for uc in user.conditions],
    }


def recent_workouts_summary(db: Session, user_id: str, limit: int = 5) -> list[dict]:
    workouts = (
        db.execute(
            select(models.Workout)
            .where(models.Workout.user_id == user_id)
            .order_by(models.Workout.started_at.desc())
            .limit(limit)
            .options(
                joinedload(models.Workout.exercises).joinedload(
                    models.WorkoutExercise.exercise
                ),
                joinedload(models.Workout.exercises).joinedload(
                    models.WorkoutExercise.sets
                ),
            )
        )
        .unique()
        .scalars()
        .all()
    )
    rows = []
    for workout in workouts:
        rows.append(
            {
                "workout_id": workout.workout_id,
                "started_at": (
                    workout.started_at.isoformat() if workout.started_at else None
                ),
                "exercise_count": len(workout.exercises),
                "set_count": sum(len(ex.sets) for ex in workout.exercises),
                "notes": workout.notes,
                "exercises": [
                    {
                        "name": ex.exercise.name,
                        "sets": [
                            {"reps": s.reps, "weight": s.weight, "rir": s.rir}
                            for s in sorted(ex.sets, key=lambda x: x.set_number)
                        ],
                    }
                    for ex in sorted(workout.exercises, key=lambda x: x.position)
                ],
            }
        )
    return rows


def recent_logs_summary(db: Session, user_id: str) -> dict:
    sleep = db.scalars(
        select(models.SleepDurationLog)
        .where(models.SleepDurationLog.user_id == user_id)
        .order_by(models.SleepDurationLog.logged_on.desc())
        .limit(7)
    ).all()
    calories = db.scalars(
        select(models.CalorieIntakeLog)
        .where(models.CalorieIntakeLog.user_id == user_id)
        .order_by(models.CalorieIntakeLog.logged_on.desc())
        .limit(7)
    ).all()
    weights = db.scalars(
        select(models.WeightLog)
        .where(models.WeightLog.user_id == user_id)
        .order_by(models.WeightLog.logged_at.desc())
        .limit(5)
    ).all()
    avg_sleep = round(sum(x.hours for x in sleep) / len(sleep), 1) if sleep else None
    avg_calories = (
        round(sum(x.calories for x in calories) / len(calories)) if calories else None
    )
    latest_weight = weights[0].weight_kg if weights else None
    delta_weight = (
        round(weights[0].weight_kg - weights[-1].weight_kg, 1)
        if len(weights) >= 2
        else 0
    )
    return {
        "avg_sleep_hours": avg_sleep,
        "avg_calories": avg_calories,
        "latest_weight": latest_weight,
        "weight_delta_recent": delta_weight,
        "sleep_entries": len(sleep),
        "calorie_entries": len(calories),
    }


def build_coach_reply(
    user: models.User, message: str, recent_workouts: list[dict], logs: dict, current_plan: dict | None = None
) -> tuple[str, str, dict]:
    llm_reply, debug = try_student_coach_reply(user, message, recent_workouts, logs, current_plan=current_plan)
    if llm_reply:
        return llm_reply, "student-llm", debug
    print(
        f"[fallback] using rule-based coach reply reason={debug.get('fallback_reason') if debug else None}"
    )
    m = message.lower()
    safety = ""
    if any(word in m for word in ["pain", "injury", "hurt", "sharp"]):
        safety = "Because you mentioned pain, keep the next session conservative, stop any movement that reproduces symptoms, and get a clinician's input if pain is sharp, worsening, or radiating. "

    goal = user.goals[0].goal.name if user.goals else "general fitness"
    adherence_note = ""
    if len(recent_workouts) < 2:
        adherence_note = "You do not have much recent training data yet, so focus on consistency first. "
    else:
        adherence_note = f"You logged {len(recent_workouts)} recent sessions, which is enough to make small adjustments but not a full overhaul. "

    sleep_note = ""
    if logs.get("avg_sleep_hours") is not None and logs["avg_sleep_hours"] < 6.5:
        sleep_note = "Your recent sleep trend is low, so keep 1–2 reps in reserve more than usual and avoid adding extra volume this week. "

    response = (
        f"{safety}{adherence_note}{sleep_note}For your {goal} goal, keep the next 7 days simple: prioritize your main sessions, stay 2–3 RIR on compounds, and only progress load when technique stays clean. "
        f"If you want, use the plan modification box to request a specific change like 'swap lunges' or 'make this a 3-day week'."
    )
    rule_context = "safety" if safety else "general"
    debug = debug or _runtime_debug_payload(
        endpoint="coach",
        selected_backend="rules",
        fallback_reason="Unknown fallback reason.",
    )
    debug["rule_context"] = rule_context
    return response.strip(), rule_context, debug


def compute_adaptation(db: Session, user: models.User, days_window: int) -> dict:
    now = datetime.utcnow()
    since = now - timedelta(days=days_window)
    workouts = (
        db.execute(
            select(models.Workout)
            .where(
                models.Workout.user_id == user.user_id,
                models.Workout.started_at >= since,
            )
            .options(
                joinedload(models.Workout.exercises).joinedload(
                    models.WorkoutExercise.sets
                )
            )
        )
        .unique()
        .scalars()
        .all()
    )
    plan = get_current_plan(db, user.user_id)
    planned_sessions = (
        len(plan.days) * max(1, math.ceil(days_window / 7)) if plan else 0
    )
    completed_sessions = len(workouts)
    adherence = (
        round((completed_sessions / planned_sessions) * 100, 1)
        if planned_sessions
        else 0.0
    )

    total_sets = sum(len(ex.sets) for w in workouts for ex in w.exercises)
    avg_rir = (
        round(
            sum(s.rir for w in workouts for ex in w.exercises for s in ex.sets)
            / max(1, total_sets),
            1,
        )
        if total_sets
        else None
    )
    logs = recent_logs_summary(db, user.user_id)

    guidance = []
    if adherence < 60:
        guidance.append(
            "Reduce complexity and aim to hit the first 2–3 sessions each week before adding more volume."
        )
    else:
        guidance.append(
            "Adherence is solid enough to keep progressing on the core lifts."
        )
    if logs.get("avg_sleep_hours") is not None and logs["avg_sleep_hours"] < 6.5:
        guidance.append(
            "Sleep is trending low, so keep effort one notch easier next week."
        )
    if avg_rir is not None and avg_rir > 4:
        guidance.append(
            "Recent sets look too easy, so you can add a small amount of load or 1 set on the first exercise each day."
        )

    structured_changes = {
        "increase_volume": bool(
            avg_rir is not None and avg_rir > 4 and adherence >= 70
        ),
        "reduce_intensity": bool(
            logs.get("avg_sleep_hours") is not None and logs["avg_sleep_hours"] < 6.5
        ),
        "suggested_days_per_week": (
            user.preference.days_per_week if user.preference else 4
        ),
    }
    return {
        "days_window": days_window,
        "completed_sessions": completed_sessions,
        "planned_sessions": planned_sessions,
        "adherence_percent": adherence,
        "avg_rir": avg_rir,
        "guidance": guidance,
        "structured_plan_changes": structured_changes,
    }


JOB_STATUS_TERMINAL = {"completed", "failed"}


def _mark_job(
    db: Session,
    job: models.PlanGenerationJob,
    *,
    status: str,
    progress: str | None = None,
    error: str | None = None,
    result_plan_id: str | None = None,
) -> None:
    job.status = status
    job.updated_at = datetime.utcnow()
    if progress is not None:
        job.progress_message = progress
    if error is not None:
        job.error_message = error
    if result_plan_id is not None:
        job.result_plan_id = result_plan_id
    if status in JOB_STATUS_TERMINAL:
        job.completed_at = datetime.utcnow()
    db.commit()


def enqueue_plan_job(
    db: Session,
    user: models.User,
    request,
    *,
    job_type: str = "generate",
    instruction: str | None = None,
) -> models.PlanGenerationJob:
    job = models.PlanGenerationJob(
        user_id=user.user_id,
        job_type=job_type,
        status="queued",
        request_payload=json.dumps(
            {
                "goal_name": getattr(request, "goal_name", None),
                "days_per_week": getattr(request, "days_per_week", None),
                "equipment": getattr(request, "equipment", None),
                "experience_level": getattr(request, "experience_level", None),
                "constraints": getattr(request, "constraints", None),
            }
        ),
        instruction=instruction,
        progress_message="Request saved. Waiting for pipeline worker.",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_plan_job(
    db: Session, job_id: str, user_id: str
) -> models.PlanGenerationJob | None:
    return db.scalar(
        select(models.PlanGenerationJob).where(
            models.PlanGenerationJob.job_id == job_id,
            models.PlanGenerationJob.user_id == user_id,
        )
    )


def get_latest_pending_plan_job(
    db: Session, user_id: str
) -> models.PlanGenerationJob | None:
    return db.scalar(
        select(models.PlanGenerationJob)
        .where(
            models.PlanGenerationJob.user_id == user_id,
            models.PlanGenerationJob.status.in_(["queued", "running"]),
        )
        .order_by(models.PlanGenerationJob.created_at.desc())
    )


def serialize_job(
    db: Session,
    job: models.PlanGenerationJob,
    *,
    latest_plan: models.WorkoutPlan | None = None,
) -> dict:
    latest_plan = latest_plan or get_current_plan(db, job.user_id)
    result_plan = None
    if job.result_plan_id:
        result_plan_model = (
            db.execute(
                select(models.WorkoutPlan)
                .where(models.WorkoutPlan.plan_id == job.result_plan_id)
                .options(
                    joinedload(models.WorkoutPlan.days)
                    .joinedload(models.PlanDay.exercises)
                    .joinedload(models.PlanExercise.sets),
                    joinedload(models.WorkoutPlan.days)
                    .joinedload(models.PlanDay.exercises)
                    .joinedload(models.PlanExercise.exercise),
                )
            )
            .unique()
            .scalars()
            .first()
        )
        if result_plan_model:
            result_plan = serialize_plan(result_plan_model)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "job_type": job.job_type,
        "progress_message": job.progress_message,
        "error_message": job.error_message,
        "latest_plan": serialize_plan(latest_plan) if latest_plan else None,
        "result_plan": result_plan,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "execution_debug": None,
    }


def process_plan_job(session_factory, job_id: str) -> None:
    db = session_factory()
    try:
        job = db.get(models.PlanGenerationJob, job_id)
        if job is None or job.status in JOB_STATUS_TERMINAL:
            return
        user = (
            db.execute(
                select(models.User)
                .where(models.User.user_id == job.user_id)
                .options(
                    joinedload(models.User.goals).joinedload(models.UserGoal.goal),
                    joinedload(models.User.preference),
                    joinedload(models.User.medical_profile),
                    joinedload(models.User.plans),
                )
            )
            .unique()
            .scalars()
            .one()
        )

        payload = json.loads(job.request_payload or "{}")
        request_like = type("Req", (), payload)()
        _mark_job(
            db,
            job,
            status="running",
            progress="Pipeline started: preprocessing profile and logs.",
        )
        time.sleep(0.4)
        _mark_job(
            db,
            job,
            status="running",
            progress="Pipeline running: generating personalized plan.",
        )
        time.sleep(0.4)

        if job.job_type == "modify":
            active_plan = get_current_plan(db, user.user_id)
            if active_plan is None:
                raise ValueError("No active plan available to modify.")
            plan, _, debug = modify_plan(db, user, active_plan, job.instruction or "")
        else:
            plan, _, debug = generate_plan(db, user, request_like)

        final_backend = (
            debug.get("selected_backend") if isinstance(debug, dict) else None
        )
        final_reason = debug.get("fallback_reason") if isinstance(debug, dict) else None
        progress = "Pipeline finished generation. Saving final structured result."
        if final_backend == "rules" and final_reason:
            progress += f" Fell back to rules: {final_reason}"
        elif final_backend == "student_model":
            progress += " Student model response was used."

        _mark_job(db, job, status="running", progress=progress)
        time.sleep(0.2)
        _mark_job(
            db,
            job,
            status="completed",
            progress="Plan ready. App can fetch and display the latest result.",
            result_plan_id=plan.plan_id,
        )
    except Exception as exc:
        job = db.get(models.PlanGenerationJob, job_id)
        if job is not None:
            _mark_job(
                db, job, status="failed", progress="Pipeline failed.", error=str(exc)
            )
    finally:
        db.close()


def _student_runtime_info() -> dict:
    runtime = get_runtime()
    info = runtime.info()
    return (
        info.to_dict()
        if hasattr(info, "to_dict")
        else {
            "available": info.available,
            "provider": info.provider,
            "base_model": info.base_model,
            "adapter_path": info.adapter_path,
            "registry_record": info.registry_record,
            "reason": info.reason,
        }
    )
