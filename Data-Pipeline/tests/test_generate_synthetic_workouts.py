from __future__ import annotations

import json


def _params() -> dict:
    return {
        "reproducibility": {"seed": 21, "hash_seed": "seed21"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-03-01",
                "num_users": 4,
                "lookback_days": 14,
                "profiles": {
                    "max_conditions_per_user": 1,
                    "max_medications_per_user": 1,
                    "max_allergies_per_user": 1,
                },
                "workouts": {
                    "min_exercises_per_plan": 2,
                    "max_exercises_per_plan": 3,
                    "sets_per_exercise": 2,
                    "workouts_per_user": 3,
                },
            }
        },
    }


def test_generate_synthetic_workouts_with_profile_inputs(tmp_path, module_loader):
    profiles_module = module_loader("generate_synthetic_profiles.py")
    workouts_module = module_loader("generate_synthetic_workouts.py")
    params = _params()

    profiles_module.generate_synthetic_profiles(
        params=params,
        output_root=tmp_path,
        run_id="RUN_PROFILES",
    )

    tables, run_dir = workouts_module.generate_synthetic_workouts(
        params=params,
        output_root=tmp_path,
        run_id="RUN_WORKOUTS",
    )

    assert run_dir.exists()
    assert (run_dir / "workout_plans.csv").exists()
    assert (run_dir / "workouts.csv").exists()

    latest = json.loads(
        (tmp_path / "synthetic_workouts" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["run_id"] == "RUN_WORKOUTS"
    assert latest["table_counts"]["workouts"] > 0

    assert len(tables["workout_plans"]) > 0
    assert len(tables["workouts"]) > 0
