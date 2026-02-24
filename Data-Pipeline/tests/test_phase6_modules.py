from pathlib import Path

from generate_synthetic_profiles import generate_synthetic_profiles
from generate_synthetic_workouts import generate_synthetic_workouts
from generate_synthetic_queries import generate_synthetic_queries
from call_teacher_llm import call_teacher_llm
from build_distillation_dataset import build_distillation_dataset
from validate_data import validate_data
from compute_stats import compute_stats
from detect_anomalies import detect_anomalies
from bias_slicing import bias_slicing


def _params() -> dict:
    return {
        "reproducibility": {"seed": 23, "hash_seed": "23"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-02-17",
                "num_users": 12,
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
        "phase6": {
            "anomaly_detection": {
                "duplicate_record_threshold": 0,
                "missing_response_threshold": 0,
                "min_response_chars": 40,
                "max_response_chars": 3000,
                "split_ratio_tolerance": 0.25,
            },
            "bias_slicing": {
                "min_group_size": 2,
                "max_mean_response_len_gap": 300,
            },
        },
    }


def test_phase6_modules_end_to_end(tmp_path: Path) -> None:
    params = _params()
    reports_root = tmp_path / "reports"

    generate_synthetic_profiles(params=params, output_root=tmp_path, run_id="profiles")
    generate_synthetic_workouts(params=params, output_root=tmp_path, run_id="workouts")
    generate_synthetic_queries(params=params, raw_root=tmp_path, run_id="queries")
    call_teacher_llm(params=params, raw_root=tmp_path, run_id="teacher")
    build_distillation_dataset(params=params, raw_root=tmp_path, run_id="distill")

    v_report, v_path = validate_data(params=params, raw_root=tmp_path, reports_root=reports_root, run_id="phase6")
    s_report, s_path = compute_stats(params=params, raw_root=tmp_path, reports_root=reports_root, run_id="phase6")
    a_report, a_path = detect_anomalies(params=params, raw_root=tmp_path, reports_root=reports_root, run_id="phase6")
    b_report, b_path = bias_slicing(params=params, raw_root=tmp_path, reports_root=reports_root, run_id="phase6")

    assert v_path.exists() and s_path.exists() and a_path.exists() and b_path.exists()
    assert v_report["valid"]
    assert s_report["num_rows"] > 0
    assert a_report["counts"]["num_rows"] > 0
    assert "bias_alert" in b_report
