from __future__ import annotations

import json


def _params() -> dict:
    return {
        "reproducibility": {"seed": 34, "hash_seed": "seed34"},
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
                    "workouts_per_user": 2,
                },
                "queries": {
                    "max_users": 3,
                    "queries_per_user_creation": 1,
                    "queries_per_user_updation": 2,
                },
            }
        },
    }


def test_generate_synthetic_queries_writes_csv_jsonl_and_latest(
    tmp_path, module_loader
):
    profiles_module = module_loader("generate_synthetic_profiles.py")
    workouts_module = module_loader("generate_synthetic_workouts.py")
    queries_module = module_loader("generate_synthetic_queries.py")
    params = _params()

    profiles_module.generate_synthetic_profiles(
        params=params,
        output_root=tmp_path,
        run_id="RUN_PROFILES",
    )
    workouts_module.generate_synthetic_workouts(
        params=params,
        output_root=tmp_path,
        run_id="RUN_WORKOUTS",
    )

    queries_df, run_dir = queries_module.generate_synthetic_queries(
        params=params,
        output_root=tmp_path,
        run_id="RUN_QUERIES",
    )

    assert run_dir.exists()
    assert (run_dir / "queries.csv").exists()
    assert (run_dir / "queries.jsonl").exists()

    expected_total = 3 * (1 + 2)
    assert len(queries_df) == expected_total
    assert set(queries_df["prompt_type"].tolist()) == {
        "plan_creation",
        "plan_updation",
    }

    latest = json.loads(
        (tmp_path / "synthetic_queries" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["run_id"] == "RUN_QUERIES"
    assert latest["total"] == expected_total
    assert latest["source_run_ids"]["synthetic_profiles"] == "RUN_PROFILES"
    assert latest["source_run_ids"]["synthetic_workouts"] == "RUN_WORKOUTS"
