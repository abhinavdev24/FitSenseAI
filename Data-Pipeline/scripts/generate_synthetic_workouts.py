"""Generate schema-aligned synthetic workout and daily health log tables."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed

EQUIPMENT = [
    ("barbell", "free_weights"),
    ("dumbbell", "free_weights"),
    ("bench", "support"),
    ("pullup_bar", "bodyweight"),
    ("cable_machine", "machine"),
    ("kettlebell", "free_weights"),
    ("resistance_band", "accessory"),
]

EXERCISES = [
    ("Barbell Back Squat", "quads"),
    ("Barbell Bench Press", "chest"),
    ("Deadlift", "posterior_chain"),
    ("Overhead Press", "shoulders"),
    ("Bent Over Row", "back"),
    ("Romanian Deadlift", "hamstrings"),
    ("Walking Lunge", "quads"),
    ("Lat Pulldown", "back"),
    ("Push Up", "chest"),
    ("Plank", "core"),
    ("Leg Press", "quads"),
    ("Seated Cable Row", "back"),
]


def _stable_uuid(kind: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"fitsense:{kind}:{value}"))


def _load_latest_profiles(raw_root: Path) -> dict[str, pd.DataFrame]:
    latest_path = raw_root / "synthetic_profiles" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            "Profiles run not found. Run Data-Pipeline/scripts/generate_synthetic_profiles.py first."
        )

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    run_dir = Path(payload["run_dir"])

    required_tables = ["users", "user_profiles", "user_goals", "calorie_targets", "sleep_targets"]
    loaded: dict[str, pd.DataFrame] = {}
    for table_name in required_tables:
        loaded[table_name] = pd.read_csv(run_dir / f"{table_name}.csv")

    return loaded


def _parse_as_of(as_of_date: str) -> date:
    return date.fromisoformat(as_of_date)


def _build_reference_catalog(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    equipment_rows = []
    for name, category in EQUIPMENT:
        equipment_rows.append(
            {
                "equipment_id": _stable_uuid("equipment", name),
                "name": name,
                "category": category,
            }
        )
    equipment_df = pd.DataFrame(equipment_rows)

    exercises_rows = []
    ex_equipment_rows = []
    equipment_ids = equipment_df["equipment_id"].tolist()

    for ex_name, primary_muscle in EXERCISES:
        exercise_id = _stable_uuid("exercise", ex_name)
        exercises_rows.append(
            {
                "exercise_id": exercise_id,
                "name": ex_name,
                "primary_muscle": primary_muscle,
                "notes": "synthetic_exercise",
            }
        )
        equipment_id = equipment_ids[int(rng.integers(0, len(equipment_ids)))]
        ex_equipment_rows.append(
            {
                "exercise_id": exercise_id,
                "equipment_id": equipment_id,
            }
        )

    return {
        "equipment": equipment_df,
        "exercises": pd.DataFrame(exercises_rows),
        "exercise_equipment": pd.DataFrame(ex_equipment_rows),
    }


def _build_plan_tables(
    users_df: pd.DataFrame,
    exercises_df: pd.DataFrame,
    as_of: date,
    min_exercises: int,
    max_exercises: int,
    sets_per_exercise: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    plans_rows = []
    plan_exercises_rows = []
    plan_sets_rows = []

    exercise_ids = exercises_df["exercise_id"].tolist()

    for user_id in users_df["user_id"].tolist():
        plan_id = _stable_uuid("plan", user_id)
        created_at = datetime.combine(as_of - timedelta(days=int(rng.integers(20, 200))), datetime.min.time()).replace(
            tzinfo=timezone.utc
        )

        plans_rows.append(
            {
                "plan_id": plan_id,
                "user_id": user_id,
                "name": "Synthetic Progressive Plan",
                "is_active": True,
                "created_at": created_at.isoformat(),
            }
        )

        count = int(rng.integers(min_exercises, max_exercises + 1))
        selected = rng.choice(exercise_ids, size=count, replace=False)

        for position, exercise_id in enumerate(selected, start=1):
            plan_exercise_id = _stable_uuid("plan_exercise", f"{plan_id}:{exercise_id}:{position}")
            plan_exercises_rows.append(
                {
                    "plan_exercise_id": plan_exercise_id,
                    "plan_id": plan_id,
                    "exercise_id": str(exercise_id),
                    "position": position,
                    "notes": "synthetic_plan_exercise",
                }
            )

            base_target_weight = round(float(rng.normal(42, 18)), 1)
            base_target_weight = min(max(base_target_weight, 10.0), 140.0)
            base_target_reps = int(rng.integers(6, 13))
            base_target_rir = int(rng.integers(1, 4))

            for set_number in range(1, sets_per_exercise + 1):
                plan_sets_rows.append(
                    {
                        "plan_set_id": _stable_uuid("plan_set", f"{plan_exercise_id}:{set_number}"),
                        "plan_exercise_id": plan_exercise_id,
                        "set_number": set_number,
                        "target_reps": base_target_reps,
                        "target_weight": base_target_weight,
                        "target_rir": base_target_rir,
                        "rest_seconds": int(rng.integers(60, 181)),
                    }
                )

    return {
        "workout_plans": pd.DataFrame(plans_rows),
        "plan_exercises": pd.DataFrame(plan_exercises_rows),
        "plan_sets": pd.DataFrame(plan_sets_rows),
    }


def _build_workout_execution_tables(
    users_df: pd.DataFrame,
    plans_df: pd.DataFrame,
    plan_exercises_df: pd.DataFrame,
    plan_sets_df: pd.DataFrame,
    as_of: date,
    workouts_per_user: int,
    lookback_days: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    workouts_rows = []
    workout_ex_rows = []
    workout_sets_rows = []

    plan_lookup = plans_df.set_index("user_id")["plan_id"].to_dict()
    plan_ex_map = plan_exercises_df.groupby("plan_id")
    plan_set_map = plan_sets_df.groupby("plan_exercise_id")

    for user_id in users_df["user_id"].tolist():
        plan_id = plan_lookup[user_id]
        user_plan_exercises = plan_ex_map.get_group(plan_id).sort_values("position")

        for session_idx in range(1, workouts_per_user + 1):
            days_ago = int(rng.integers(1, lookback_days + 1))
            start_dt = datetime.combine(as_of - timedelta(days=days_ago), datetime.min.time()).replace(tzinfo=timezone.utc)
            start_dt = start_dt + timedelta(hours=int(rng.integers(5, 21)), minutes=int(rng.integers(0, 60)))
            duration = int(rng.integers(35, 95))
            end_dt = start_dt + timedelta(minutes=duration)

            workout_id = _stable_uuid("workout", f"{user_id}:{session_idx}")
            workouts_rows.append(
                {
                    "workout_id": workout_id,
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "started_at": start_dt.isoformat(),
                    "ended_at": end_dt.isoformat(),
                    "notes": "synthetic_workout",
                }
            )

            workout_ex_position = 1
            for row in user_plan_exercises.itertuples(index=False):
                workout_exercise_id = _stable_uuid("workout_exercise", f"{workout_id}:{row.plan_exercise_id}")
                workout_ex_rows.append(
                    {
                        "workout_exercise_id": workout_exercise_id,
                        "workout_id": workout_id,
                        "exercise_id": row.exercise_id,
                        "plan_exercise_id": row.plan_exercise_id,
                        "position": workout_ex_position,
                        "notes": "synthetic_workout_exercise",
                    }
                )

                target_sets = plan_set_map.get_group(row.plan_exercise_id).sort_values("set_number")
                for plan_set in target_sets.itertuples(index=False):
                    reps = int(np.clip(plan_set.target_reps + rng.integers(-2, 3), 1, 20))
                    weight = round(float(np.clip(plan_set.target_weight + rng.normal(0, 4), 2.0, 250.0)), 1)
                    rir = int(np.clip(plan_set.target_rir + rng.integers(-1, 2), 0, 5))
                    completed_at = start_dt + timedelta(minutes=3 * workout_ex_position + plan_set.set_number)

                    workout_sets_rows.append(
                        {
                            "workout_set_id": _stable_uuid(
                                "workout_set", f"{workout_exercise_id}:{plan_set.set_number}"
                            ),
                            "workout_exercise_id": workout_exercise_id,
                            "set_number": int(plan_set.set_number),
                            "reps": reps,
                            "weight": weight,
                            "rir": rir,
                            "is_warmup": bool(plan_set.set_number == 1 and rng.random() < 0.25),
                            "completed_at": completed_at.isoformat(),
                        }
                    )

                workout_ex_position += 1

    workouts_df = pd.DataFrame(workouts_rows).sort_values(["user_id", "started_at"]).reset_index(drop=True)
    return {
        "workouts": workouts_df,
        "workout_exercises": pd.DataFrame(workout_ex_rows),
        "workout_sets": pd.DataFrame(workout_sets_rows),
    }


def _build_daily_logs(
    users_df: pd.DataFrame,
    user_goals_df: pd.DataFrame,
    calorie_targets_df: pd.DataFrame,
    sleep_targets_df: pd.DataFrame,
    as_of: date,
    lookback_days: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    primary_goal = (
        user_goals_df.sort_values(["user_id", "priority"]).drop_duplicates("user_id").set_index("user_id")["goal_id"].to_dict()
    )
    calorie_target_map = calorie_targets_df.set_index("user_id")["maintenance_calories"].to_dict()
    sleep_target_map = sleep_targets_df.set_index("user_id")["target_sleep_hours"].to_dict()

    calorie_rows = []
    sleep_rows = []
    weight_rows = []

    for user_id in users_df["user_id"].tolist():
        maintenance = int(calorie_target_map[user_id])
        sleep_target = float(sleep_target_map[user_id])
        goal = primary_goal.get(user_id)

        goal_trend_per_week = 0.0
        if goal is not None:
            goal_str = str(goal)
            if "fat_loss" in goal_str:
                goal_trend_per_week = -0.2
            elif "muscle_gain" in goal_str:
                goal_trend_per_week = 0.12

        initial_weight = float(np.clip(rng.normal(76, 14), 45, 150))

        sampled_days = sorted(rng.choice(np.arange(lookback_days), size=max(10, int(lookback_days * 0.7)), replace=False).tolist())
        for d in sampled_days:
            day = as_of - timedelta(days=int(d))

            calorie_rows.append(
                {
                    "calorie_log_id": _stable_uuid("calorie_log", f"{user_id}:{day.isoformat()}"),
                    "user_id": user_id,
                    "log_date": day.isoformat(),
                    "calories_consumed": int(np.clip(maintenance + rng.normal(0, 220), 900, 5000)),
                    "notes": "synthetic_calorie_log",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            sleep_rows.append(
                {
                    "sleep_log_id": _stable_uuid("sleep_log", f"{user_id}:{day.isoformat()}"),
                    "user_id": user_id,
                    "log_date": day.isoformat(),
                    "sleep_duration_hours": round(float(np.clip(sleep_target + rng.normal(0, 0.8), 3.5, 12.0)), 2),
                    "notes": "synthetic_sleep_log",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        for week in range(0, lookback_days + 1, 7):
            day = as_of - timedelta(days=week)
            trend = goal_trend_per_week * (week / 7.0)
            weight = float(np.clip(initial_weight - trend + rng.normal(0, 0.35), 40, 180))
            bf = float(np.clip(rng.normal(24, 6), 6, 45))

            weight_rows.append(
                {
                    "weight_log_id": _stable_uuid("weight_log", f"{user_id}:{day.isoformat()}"),
                    "user_id": user_id,
                    "logged_at": datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat(),
                    "weight_kg": round(weight, 2),
                    "body_fat_percentage": round(bf, 2),
                    "notes": "synthetic_weight_log",
                }
            )

    return {
        "calorie_intake_logs": pd.DataFrame(calorie_rows),
        "sleep_duration_logs": pd.DataFrame(sleep_rows),
        "weight_logs": pd.DataFrame(weight_rows),
    }


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, df in tables.items():
        df.to_csv(output_dir / f"{table_name}.csv", index=False)


def generate_synthetic_workouts(params: dict, output_root: Path, run_id: str | None = None) -> tuple[dict[str, pd.DataFrame], Path]:
    synthetic_cfg = params["phase2"]["synthetic"]
    workout_cfg = synthetic_cfg["workouts"]

    seed = int(params["reproducibility"]["seed"]) + 1
    rng = np.random.default_rng(seed)

    as_of = _parse_as_of(str(synthetic_cfg["as_of_date"]))
    lookback_days = int(synthetic_cfg["lookback_days"])

    profiles_data = _load_latest_profiles(output_root)
    users_df = profiles_data["users"]

    catalog_tables = _build_reference_catalog(rng=rng)
    plan_tables = _build_plan_tables(
        users_df=users_df,
        exercises_df=catalog_tables["exercises"],
        as_of=as_of,
        min_exercises=int(workout_cfg["min_exercises_per_plan"]),
        max_exercises=int(workout_cfg["max_exercises_per_plan"]),
        sets_per_exercise=int(workout_cfg["sets_per_exercise"]),
        rng=rng,
    )
    execution_tables = _build_workout_execution_tables(
        users_df=users_df,
        plans_df=plan_tables["workout_plans"],
        plan_exercises_df=plan_tables["plan_exercises"],
        plan_sets_df=plan_tables["plan_sets"],
        as_of=as_of,
        workouts_per_user=int(workout_cfg["workouts_per_user"]),
        lookback_days=lookback_days,
        rng=rng,
    )
    daily_tables = _build_daily_logs(
        users_df=users_df,
        user_goals_df=profiles_data["user_goals"],
        calorie_targets_df=profiles_data["calorie_targets"],
        sleep_targets_df=profiles_data["sleep_targets"],
        as_of=as_of,
        lookback_days=lookback_days,
        rng=rng,
    )

    tables: dict[str, pd.DataFrame] = {}
    tables.update(catalog_tables)
    tables.update(plan_tables)
    tables.update(execution_tables)
    tables.update(daily_tables)

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = output_root / "synthetic_workouts" / run_id
    _write_tables(tables=tables, output_dir=run_dir)

    latest_path = output_root / "synthetic_workouts" / "latest.json"
    latest_payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "seed": seed,
        "as_of_date": str(as_of),
        "table_counts": {name: int(len(df)) for name, df in tables.items()},
    }
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return tables, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic workouts dataset")
    parser.add_argument("--params", default="Data-Pipeline/params.yaml", help="Path to params.yaml")
    parser.add_argument("--output-root", default=None, help="Optional output root override")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(int(params["reproducibility"]["seed"]), str(params["reproducibility"]["hash_seed"]))

    output_root = Path(args.output_root) if args.output_root else Path(str(params["paths"]["raw_data_dir"]))
    logger = setup_logger(
        name="fitsense.synthetic_workouts",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    _, run_dir = generate_synthetic_workouts(params=params, output_root=output_root, run_id=args.run_id)
    logger.info("Generated synthetic workout data at %s", run_dir)


if __name__ == "__main__":
    main()
