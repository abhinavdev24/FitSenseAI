from pathlib import Path

from generate_synthetic_profiles import generate_synthetic_profiles
from generate_synthetic_workouts import generate_synthetic_workouts
from generate_synthetic_queries import generate_synthetic_queries
from call_teacher_llm import call_teacher_llm


def _params() -> dict:
    return {
        "reproducibility": {"seed": 13, "hash_seed": "13"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-02-17",
                "num_users": 6,
                "lookback_days": 21,
                "profiles": {
                    "max_conditions_per_user": 2,
                    "max_medications_per_user": 1,
                    "max_allergies_per_user": 1,
                },
                "workouts": {
                    "workouts_per_user": 2,
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
        "phase4": {
            "teacher_llm": {
                "provider": "mock",
                "model_name": "teacher-mock-v1",
                "endpoint_url": "",
                "api_key_env": "OPENAI_API_KEY",
                "timeout_seconds": 10,
                "max_retries": 2,
                "retry_backoff_seconds": 0.1,
                "temperature": 0.2,
                "top_p": 1.0,
                "max_output_tokens": 256,
                "max_queries": None,
            }
        },
    }


def test_call_teacher_llm_mock_mode(tmp_path: Path) -> None:
    params = _params()

    generate_synthetic_profiles(params=params, output_root=tmp_path, run_id="profiles")
    generate_synthetic_workouts(params=params, output_root=tmp_path, run_id="workouts")
    queries_df, _ = generate_synthetic_queries(params=params, raw_root=tmp_path, run_id="queries")

    records, out_dir = call_teacher_llm(params=params, raw_root=tmp_path, run_id="teacher")

    assert out_dir.exists()
    assert (out_dir / "responses.jsonl").exists()
    assert (out_dir / "responses.csv").exists()
    assert (out_dir / "summary.json").exists()

    assert len(records) == len(queries_df)
    assert all(r["status"] == "success" for r in records)
    assert all(r["response_text"] for r in records)

    sample = records[0]
    assert sample["query_id"]
    assert sample["scenario_id"]
    assert sample["provider"] == "mock"
    assert sample["model_name"] == "teacher-mock-v1"
    assert isinstance(sample["post_validation"], dict)
    assert "is_valid" in sample["post_validation"]
