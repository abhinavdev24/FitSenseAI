from pathlib import Path

from generate_synthetic_profiles import generate_synthetic_profiles
from generate_synthetic_workouts import generate_synthetic_workouts
from generate_synthetic_queries import generate_synthetic_queries


def _params() -> dict:
    return {
        "reproducibility": {"seed": 9, "hash_seed": "9"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-02-17",
                "num_users": 10,
                "lookback_days": 28,
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
        "phase3": {
            "synthetic_queries": {
                "prompts_per_type": 1,
                "prompt_types": [
                    "plan_creation",
                    "plan_modification",
                    "safety_adjustment",
                    "progress_adaptation",
                ],
            }
        },
    }


def test_generate_queries_coverage_and_metadata(tmp_path: Path) -> None:
    params = _params()
    generate_synthetic_profiles(params=params, output_root=tmp_path, run_id="profiles")
    generate_synthetic_workouts(params=params, output_root=tmp_path, run_id="workouts")

    queries_df, run_dir = generate_synthetic_queries(params=params, raw_root=tmp_path, run_id="queries")

    assert run_dir.exists()
    assert (run_dir / "queries.jsonl").exists()
    assert (run_dir / "queries.csv").exists()

    expected_prompt_types = set(params["phase3"]["synthetic_queries"]["prompt_types"])
    assert set(queries_df["prompt_type"].unique()) == expected_prompt_types

    counts = queries_df.groupby("user_id")["prompt_type"].nunique()
    assert (counts == len(expected_prompt_types)).all()

    required_cols = {
        "query_id",
        "scenario_id",
        "user_id",
        "prompt_type",
        "prompt_text",
        "slice_tags",
        "expected_safety_constraints",
        "context_summary",
        "source_run_ids",
    }
    assert required_cols.issubset(set(queries_df.columns))

    for row in queries_df.itertuples(index=False):
        assert isinstance(row.slice_tags, dict)
        assert isinstance(row.expected_safety_constraints, list)
        assert isinstance(row.context_summary, dict)
        assert isinstance(row.source_run_ids, dict)

        assert row.slice_tags["age_band"]
        assert row.slice_tags["goal_type"]
        assert row.slice_tags["activity_level"]
        assert row.prompt_text
        assert len(row.expected_safety_constraints) > 0
