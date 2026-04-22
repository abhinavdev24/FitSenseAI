"""Tests for Model-Pipeline/scripts/prepare_training_data.py.

Covers _validate_record() and convert_record() — both are pure data-transformation
functions that require no GPU and no external API calls.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides: Any) -> dict[str, Any]:
    """Build a fully valid raw teacher record, with optional field overrides."""
    record: dict[str, Any] = {
        "status": "success",
        "response_id": "resp-001",
        "query_id": "qid-001",
        "prompt_type": "plan_creation",
        "model_name": "qwen3:32b",
        "request_payload": {
            "messages": [
                {"role": "system", "content": "You are a fitness coach."},
                {"role": "user", "content": "Create a 3-day workout plan."},
            ]
        },
        "response_text": '{"tool_name": "log_workout", "tool_input": {}}',
        "response_json": {"tool_name": "log_workout", "tool_input": {}},
        "raw_response": {
            "choices": [
                {
                    "message": {
                        "reasoning": "The user wants a workout plan, so I will call log_workout.",
                        "content": '{"tool_name": "log_workout", "tool_input": {}}',
                    }
                }
            ]
        },
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Tests for _validate_record
# ---------------------------------------------------------------------------

class TestValidateRecord:
    def test_valid_record_has_no_issues(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        issues = mod._validate_record(_make_record())
        assert issues == []

    def test_bad_status_reported(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        record = _make_record(status="error")
        issues = mod._validate_record(record)
        assert any("status" in issue for issue in issues)

    def test_missing_response_json_reported(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        record = _make_record(response_json=None)
        issues = mod._validate_record(record)
        assert any("response_json" in issue for issue in issues)

    def test_empty_response_text_reported(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        record = _make_record(response_text="   ")
        issues = mod._validate_record(record)
        assert any("response_text" in issue for issue in issues)

    def test_missing_reasoning_reported(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        record = _make_record(
            raw_response={"choices": [{"message": {"reasoning": None, "content": "x"}}]}
        )
        issues = mod._validate_record(record)
        assert any("reasoning" in issue for issue in issues)

    def test_too_few_request_messages_reported(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        record = _make_record(
            request_payload={"messages": [{"role": "system", "content": "sys"}]}
        )
        issues = mod._validate_record(record)
        assert any("message" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Tests for convert_record
# ---------------------------------------------------------------------------

class TestConvertRecord:
    def test_valid_record_returns_three_message_dict(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        result = mod.convert_record(_make_record())

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"

    def test_assistant_content_has_think_block(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        result = mod.convert_record(_make_record())

        assert result is not None
        assistant = result["messages"][2]["content"]
        assert assistant.startswith("<think>")
        assert "</think>" in assistant

    def test_residual_think_in_response_text_is_stripped(self, module_loader: Any) -> None:
        """If response_text already contains a <think> block it must be removed."""
        mod = module_loader("scripts/prepare_training_data.py")
        record = _make_record(
            response_text="<think>\nstale thought\n</think>\n{\"tool_name\": \"log_workout\"}"
        )
        result = mod.convert_record(record)

        assert result is not None
        assistant = result["messages"][2]["content"]
        # The stale <think> from response_text should be gone; only the one from
        # raw_response reasoning should remain (as the outer <think> block).
        assert assistant.count("<think>") == 1

    def test_invalid_record_returns_none(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        record = _make_record(status="failed")
        result = mod.convert_record(record)
        assert result is None

    def test_metadata_fields_are_preserved(self, module_loader: Any) -> None:
        mod = module_loader("scripts/prepare_training_data.py")
        result = mod.convert_record(_make_record())

        assert result is not None
        meta = result["metadata"]
        assert meta["response_id"] == "resp-001"
        assert meta["query_id"] == "qid-001"
        assert meta["prompt_type"] == "plan_creation"
