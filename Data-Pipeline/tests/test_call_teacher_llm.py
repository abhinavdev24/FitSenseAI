"""Tests for call_teacher_llm.py.

Covers:
- Full pipeline integration using a monkeypatched Groq client so no real HTTP
  calls are made.  The fake responder inspects the messages to choose a
  prompt-type-appropriate JSON reply that satisfies _post_validate().
- _load_env_file_if_present() behaviour for various .env file locations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from call_teacher_llm import call_teacher_llm
from generate_synthetic_profiles import generate_synthetic_profiles
from generate_synthetic_queries import generate_synthetic_queries
from generate_synthetic_workouts import generate_synthetic_workouts


# ---------------------------------------------------------------------------
# Helpers: fake HTTP layer
# ---------------------------------------------------------------------------


def _fake_response_body(prompt_type: str) -> dict:
    """Return a minimal valid coaching response dict for the given prompt_type."""
    bodies: dict = {
        "plan_creation": {
            "message": "Here is your 7-day muscle gain plan.",
            "data": {
                "name": "7-Day Muscle Gain",
                "exercises": [
                    {
                        "exercise_name": "Goblet Squat",
                        "sets": 3,
                        "reps": 10,
                        "weight_kg": 20.0,
                        "rir": 2,
                        "rest_seconds": 90,
                    }
                ],
            },
        },
        "plan_modification": {
            "message": "I've updated your plan based on recent performance.",
            "data": {
                "modifications": [
                    {
                        "action": "update",
                        "exercise_name": "Goblet Squat",
                        "updates": {"weight_kg": 22.5},
                    }
                ]
            },
        },
        "safety_adjustment": {
            "message": "I've identified some movements to modify for your lower back.",
            "data": {
                "safe": False,
                "reason": "High spinal load contraindicated with lower_back_pain.",
                "alternatives": [
                    {
                        "exercise_name": "Hip Thrust",
                        "reason": "Low spinal load glute exercise.",
                    }
                ],
            },
        },
        "progress_adaptation": {
            "message": "You've been improving — time to add some volume.",
            "data": {
                "volume_trend": "improving",
                "strength_trend": "stable",
                "adherence_rate": 0.92,
                "recommendations": ["Add one set per compound movement next week."],
            },
        },
        "progress_comment": {
            "message": "You're on track with your muscle gain goal.",
            "data": {
                "summary": "12 sessions completed with consistent progressive overload.",
                "next_focus": "Increase weekly volume by 10%.",
            },
        },
        "workout_logging": {
            "message": "Workout logged successfully. Good session!",
            "data": {
                "exercises": [
                    {
                        "exercise_name": "Bench Press",
                        "sets": [
                            {"set_number": 1, "reps": 10, "weight_kg": 62.5, "rir": 2}
                        ],
                    }
                ]
            },
        },
        "metric_logging": {
            "message": "Weight logged for today.",
            "data": {"metric": "weight", "value": 75.0, "date": "2026-03-07"},
        },
        "coaching_qa": {
            "message": "For hypertrophy, 8-12 reps per set at 65-80% of 1RM is optimal.",
            "data": None,
        },
    }
    return bodies.get(prompt_type, {"message": "Done.", "data": None})


def _detect_prompt_type(user_msg: str) -> str:
    """Extract action type from [ACTION: ...] tag in the user message."""
    for ptype in [
        "plan_creation",
        "plan_modification",
        "safety_adjustment",
        "progress_adaptation",
        "progress_comment",
        "workout_logging",
        "metric_logging",
    ]:
        if f"[ACTION: {ptype}]" in user_msg:
            return ptype
    return "coaching_qa"


def _make_groq_response(prompt_type: str) -> MagicMock:
    """Build a mock Groq ChatCompletion response for the given prompt_type."""
    body = _fake_response_body(prompt_type)

    mock_message = MagicMock()
    mock_message.content = json.dumps(body)

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model_dump.return_value = {"id": "fake-id", "object": "chat.completion"}

    return mock_response


# ---------------------------------------------------------------------------
# Params fixture
# ---------------------------------------------------------------------------


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
                "provider": "groq",
                "model_name": "teacher-test-v1",
                "endpoint_url": "https://fake-endpoint.example.com/v1/chat/completions",
                "api_key_env": "GROQ_API_KEY",
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


# ---------------------------------------------------------------------------
# Integration test: full pipeline with monkeypatched HTTP
# ---------------------------------------------------------------------------


def test_call_teacher_llm_openai_compatible(tmp_path: Path, monkeypatch) -> None:
    """Full pipeline test: profiles → workouts → queries → teacher LLM responses.

    The Groq client is monkeypatched so no real HTTP calls occur.  The fake
    responder inspects the messages to return a prompt-type-appropriate JSON
    reply that satisfies _post_validate().
    """
    params = _params()

    # Provide a fake API key so the code does not raise NonRetriableTeacherError.
    monkeypatch.setenv("GROQ_API_KEY", "test-key-fake")

    # Bypass filesystem lookup for the coach system prompt.
    monkeypatch.setattr(
        "call_teacher_llm._build_system_prompt",
        lambda: "You are FitSenseAI. Respond ONLY with valid JSON.",
    )

    # ------------------------------------------------------------------
    # Build upstream pipeline state inside tmp_path
    # ------------------------------------------------------------------
    generate_synthetic_profiles(params=params, output_root=tmp_path, run_id="profiles")
    generate_synthetic_workouts(params=params, output_root=tmp_path, run_id="workouts")
    queries_df, _ = generate_synthetic_queries(
        params=params, raw_root=tmp_path, run_id="queries"
    )

    # ------------------------------------------------------------------
    # Monkeypatch the Groq client
    # ------------------------------------------------------------------
    def fake_create(**kwargs) -> MagicMock:
        messages = kwargs.get("messages", [])
        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        return _make_groq_response(_detect_prompt_type(user_msg))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("call_teacher_llm.Groq", return_value=mock_client):
        records, out_dir = call_teacher_llm(
            params=params, raw_root=tmp_path, run_id="teacher"
        )

    # ------------------------------------------------------------------
    # Structural assertions
    # ------------------------------------------------------------------
    assert out_dir.exists(), "Output directory was not created"
    assert (out_dir / "responses.jsonl").exists(), "responses.jsonl missing"
    assert (out_dir / "responses.csv").exists(), "responses.csv missing"
    assert (out_dir / "summary.json").exists(), "summary.json missing"

    # One record per query
    assert len(records) == len(queries_df)

    # Every call must have succeeded
    assert all(r["status"] == "success" for r in records), (
        "Some records have non-success status: "
        + str([r for r in records if r["status"] != "success"])
    )

    # response_json must be a dict with the expected top-level keys
    assert all(isinstance(r["response_json"], dict) for r in records)
    assert all("message" in r["response_json"] for r in records)
    assert all("data" in r["response_json"] for r in records)

    # Post-validation must pass for every record
    assert all(r["post_validation"]["is_valid"] for r in records), (
        "Some records failed post_validation: "
        + str([r for r in records if not r["post_validation"]["is_valid"]])
    )

    # Spot-check a single record for expected metadata fields
    sample = records[0]
    assert sample["query_id"]
    assert sample["scenario_id"]
    assert sample["provider"] == "groq"
    assert sample["model_name"] == "teacher-test-v1"
    assert isinstance(sample["post_validation"], dict)
    assert "is_valid" in sample["post_validation"]


# ---------------------------------------------------------------------------
# Unit test: _load_env_file_if_present
# ---------------------------------------------------------------------------


def test_load_env_file(tmp_path: Path, monkeypatch) -> None:
    """The loader should read variables from .env/.env.local in expected locations."""
    # create a fake repo structure under tmp_path
    repo_root = tmp_path / "repo"
    (repo_root / "Data-Pipeline" / "scripts").mkdir(parents=True)

    # write .env in repo root
    env_file = repo_root / ".env"
    env_file.write_text("GROQ_API_KEY=mysecret\n# comment line\n")

    # ensure variable is not already present
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    from call_teacher_llm import _load_env_file_if_present

    _load_env_file_if_present(repo_root=repo_root)
    assert os.getenv("GROQ_API_KEY") == "mysecret"

    # also check Data-Pipeline/.env takes precedence if repo root missing
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env_file.unlink()
    alt_file = repo_root / "Data-Pipeline" / ".env"
    alt_file.write_text("GROQ_API_KEY=another\n")
    _load_env_file_if_present(repo_root=repo_root)
    assert os.getenv("GROQ_API_KEY") == "another"

    # ensure scripts/.env.local is picked up when parent files are absent
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    alt_file.unlink()
    script_env = repo_root / "Data-Pipeline" / "scripts" / ".env.local"
    script_env.write_text("GROQ_API_KEY=scripts_local\n")
    _load_env_file_if_present(repo_root=repo_root)
    assert os.getenv("GROQ_API_KEY") == "scripts_local"
