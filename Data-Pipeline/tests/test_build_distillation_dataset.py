from pathlib import Path

from generate_synthetic_profiles import generate_synthetic_profiles
from generate_synthetic_workouts import generate_synthetic_workouts
from generate_synthetic_queries import generate_synthetic_queries
from call_teacher_llm import call_teacher_llm
from build_distillation_dataset import build_distillation_dataset


def _params() -> dict:
    return {
        "reproducibility": {"seed": 17, "hash_seed": "17"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-02-17",
                "num_users": 8,
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
        "phase5": {
            "distillation": {
                "min_response_chars": 40,
                "require_post_validation": True,
                "reject_on_safety_flags": True,
                "split": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
            }
        },
    }


def test_build_distillation_dataset_end_to_end(tmp_path: Path) -> None:
    params = _params()

    generate_synthetic_profiles(params=params, output_root=tmp_path, run_id="profiles")
    generate_synthetic_workouts(params=params, output_root=tmp_path, run_id="workouts")
    queries_df, _ = generate_synthetic_queries(params=params, raw_root=tmp_path, run_id="queries")
    teacher_records, _ = call_teacher_llm(params=params, raw_root=tmp_path, run_id="teacher")

    distill_df, out_dir = build_distillation_dataset(params=params, raw_root=tmp_path, run_id="distill")

    assert out_dir.exists()
    assert (out_dir / "all_records.jsonl").exists()
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    assert (out_dir / "summary.json").exists()

    assert len(teacher_records) == len(queries_df)
    assert len(distill_df) > 0

    assert set(distill_df["split"].unique()) <= {"train", "val", "test"}
    assert distill_df["record_id"].is_unique

    train_ids = set(distill_df.loc[distill_df["split"] == "train", "record_id"])
    val_ids = set(distill_df.loc[distill_df["split"] == "val", "record_id"])
    test_ids = set(distill_df.loc[distill_df["split"] == "test", "record_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    sample = distill_df.iloc[0]
    assert sample["instruction"]
    assert sample["response"]
    assert isinstance(sample["context"], dict)
    assert isinstance(sample["metadata"], dict)
