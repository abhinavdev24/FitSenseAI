from pathlib import Path

from generate_synthetic_profiles import generate_synthetic_profiles


def _params() -> dict:
    return {
        "reproducibility": {"seed": 42, "hash_seed": "42"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-02-17",
                "num_users": 25,
                "lookback_days": 30,
                "profiles": {
                    "max_conditions_per_user": 2,
                    "max_medications_per_user": 2,
                    "max_allergies_per_user": 2,
                },
                "workouts": {
                    "workouts_per_user": 4,
                    "min_exercises_per_plan": 3,
                    "max_exercises_per_plan": 5,
                    "sets_per_exercise": 3,
                },
            }
        },
    }


def test_generate_profiles_schema_alignment(tmp_path: Path) -> None:
    tables, run_dir = generate_synthetic_profiles(params=_params(), output_root=tmp_path, run_id="test_run")

    assert run_dir.exists()
    users = tables["users"]
    profiles = tables["user_profiles"]
    user_goals = tables["user_goals"]
    goals = tables["goals"]
    conditions = tables["conditions"]
    user_conditions = tables["user_conditions"]
    calorie_targets = tables["calorie_targets"]
    sleep_targets = tables["sleep_targets"]

    assert len(users) == 25
    assert users["user_id"].is_unique
    assert set(profiles["user_id"]) == set(users["user_id"])
    assert set(user_goals["user_id"]).issubset(set(users["user_id"]))
    assert set(user_goals["goal_id"]).issubset(set(goals["goal_id"]))
    assert set(user_conditions["condition_id"]).issubset(set(conditions["condition_id"]))
    assert set(calorie_targets["user_id"]) == set(users["user_id"])
    assert set(sleep_targets["user_id"]) == set(users["user_id"])

    assert profiles["height_cm"].between(145.0, 205.0).all()
    assert calorie_targets["maintenance_calories"].between(1400, 3800).all()
    assert sleep_targets["target_sleep_hours"].between(6.0, 9.5).all()
