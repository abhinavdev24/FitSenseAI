from pathlib import Path

from generate_synthetic_profiles import generate_synthetic_profiles
from generate_synthetic_workouts import generate_synthetic_workouts


def _params() -> dict:
    return {
        "reproducibility": {"seed": 7, "hash_seed": "7"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-02-17",
                "num_users": 12,
                "lookback_days": 35,
                "profiles": {
                    "max_conditions_per_user": 2,
                    "max_medications_per_user": 1,
                    "max_allergies_per_user": 1,
                },
                "workouts": {
                    "workouts_per_user": 3,
                    "min_exercises_per_plan": 3,
                    "max_exercises_per_plan": 4,
                    "sets_per_exercise": 3,
                },
            }
        },
    }


def test_generate_workouts_with_fk_consistency(tmp_path: Path) -> None:
    params = _params()
    generate_synthetic_profiles(params=params, output_root=tmp_path, run_id="profiles_run")
    tables, run_dir = generate_synthetic_workouts(params=params, output_root=tmp_path, run_id="workouts_run")

    assert run_dir.exists()

    workouts = tables["workouts"]
    workout_exercises = tables["workout_exercises"]
    workout_sets = tables["workout_sets"]
    plans = tables["workout_plans"]
    plan_exercises = tables["plan_exercises"]
    plan_sets = tables["plan_sets"]
    calorie_logs = tables["calorie_intake_logs"]
    sleep_logs = tables["sleep_duration_logs"]
    weight_logs = tables["weight_logs"]

    assert set(workouts["plan_id"]).issubset(set(plans["plan_id"]))
    assert set(workout_exercises["workout_id"]).issubset(set(workouts["workout_id"]))
    assert set(workout_exercises["plan_exercise_id"]).issubset(set(plan_exercises["plan_exercise_id"]))
    assert set(workout_sets["workout_exercise_id"]).issubset(set(workout_exercises["workout_exercise_id"]))
    assert set(plan_sets["plan_exercise_id"]).issubset(set(plan_exercises["plan_exercise_id"]))

    assert workout_sets["reps"].between(1, 20).all()
    assert workout_sets["weight"].between(2.0, 250.0).all()
    assert workout_sets["rir"].between(0, 5).all()
    assert calorie_logs["calories_consumed"].between(900, 5000).all()
    assert sleep_logs["sleep_duration_hours"].between(3.5, 12.0).all()
    assert weight_logs["weight_kg"].between(40.0, 180.0).all()

    assert not calorie_logs.duplicated(subset=["user_id", "log_date"]).any()
    assert not sleep_logs.duplicated(subset=["user_id", "log_date"]).any()
