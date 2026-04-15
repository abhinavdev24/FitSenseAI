"""Unit tests for the LLM runtime module."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def runtime():
    """Create a fresh StudentLLMRuntime instance."""
    from app.llm_runtime import get_runtime
    get_runtime.cache_clear()
    return get_runtime()


class TestCallOpenAICompatible:
    """Tests for _call_openai_compatible method."""

    def test_returns_none_when_api_key_missing(self, runtime):
        """Should return None when OPENAI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime._call_openai_compatible(
                system_prompt="Test system",
                user_message="Test user",
                max_new_tokens=100,
            )
            assert result is None

    def test_returns_none_when_api_url_missing(self, runtime):
        """Should return None when OPENAI_API_URL is not set."""
        env = {"OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime._call_openai_compatible(
                system_prompt="Test system",
                user_message="Test user",
                max_new_tokens=100,
            )
            assert result is None

    @patch("requests.post")
    def test_successful_api_call(self, mock_post: MagicMock, runtime):
        """Should return content from successful API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
            "OPENAI_MODEL": "test-model",
        }
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime._call_openai_compatible(
                system_prompt="Test system",
                user_message="Test user",
                max_new_tokens=100,
            )

            assert result == "Generated text"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
            assert call_kwargs["json"]["model"] == "test-model"
            assert call_kwargs["timeout"] == 120

    @patch("requests.post")
    def test_api_call_failure_returns_none(self, mock_post: MagicMock, runtime):
        """Should return None and print warning on API failure."""
        mock_post.side_effect = Exception("Connection error")

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime._call_openai_compatible(
                system_prompt="Test system",
                user_message="Test user",
                max_new_tokens=100,
            )

            assert result is None

    @patch("requests.post")
    def test_default_model_used_when_not_specified(self, mock_post: MagicMock, runtime):
        """Should use default model when OPENAI_MODEL is not set."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            runtime._call_openai_compatible(
                system_prompt="Test",
                user_message="Test",
                max_new_tokens=100,
            )

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "qwen/qwen3-8b:free"


class TestJSONExtraction:
    """Tests for JSON extraction and repair methods."""

    def test_extract_first_json_object_valid(self, runtime):
        """Should extract valid JSON object from text."""
        text = 'Some text before {"key": "value", "nested": {"inner": 123}} some text after'
        result = runtime._extract_first_json_object(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["nested"]["inner"] == 123

    def test_extract_first_json_object_empty_input(self, runtime):
        """Should return None for empty input."""
        assert runtime._extract_first_json_object("") is None
        assert runtime._extract_first_json_object(None) is None

    def test_extract_first_json_object_no_json(self, runtime):
        """Should return None when no JSON object in text."""
        assert runtime._extract_first_json_object("No JSON here") is None

    def test_repair_common_json_issues_markdown_fence(self, runtime):
        """Should remove markdown code fences."""
        text = '```json\n{"key": "value"}\n```'
        result = runtime._repair_common_json_issues(text)
        assert result == '{"key": "value"}'

    def test_repair_common_json_issues_trailing_comma(self, runtime):
        """Should remove trailing commas before closing braces/brackets."""
        text = '{"key": "value", "key2": 123, }'
        result = runtime._repair_common_json_issues(text)
        assert result == '{"key": "value", "key2": 123 }'

    def test_repair_common_json_issues_case_insensitive(self, runtime):
        """Should handle case-insensitive markdown fences."""
        text = '```JSON\n{"key": "value"}\n```'
        result = runtime._repair_common_json_issues(text)
        assert result == '{"key": "value"}'


class TestGenerateText:
    """Tests for generate_text method and fallback chain."""

    @patch("requests.post")
    def test_generate_text_uses_openai_api(self, mock_post: MagicMock, runtime):
        """Should use OpenAI-compatible API when configured."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated response text"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime.generate_text(
                system_prompt="Test system",
                user_message="Test user",
                max_new_tokens=100,
            )

            assert result == "Generated response text"
            mock_post.assert_called_once()

    @patch("requests.post")
    def test_generate_text_openai_api_failure(self, mock_post: MagicMock, runtime):
        """Should return None when OpenAI API fails and no cloud/local configured."""
        mock_post.side_effect = Exception("API Error")

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime.generate_text(
                system_prompt="Test",
                user_message="Test",
                max_new_tokens=100,
            )
            # Without cloud or local model, returns None after OpenAI failure
            assert result is None


class TestGeneratePlanJSON:
    """Tests for generate_plan_json method."""

    @patch("requests.post")
    def test_generate_plan_json_extracts_json_from_response(self, mock_post: MagicMock, runtime):
        """Should extract and parse JSON from API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"plan_name": "Test Plan", "days": [{"name": "Day 1", "day_order": 1, "exercises": []}]}'
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime.generate_plan_json(user_message="Generate a plan")

            assert result is not None
            assert result["plan_name"] == "Test Plan"
            assert len(result["days"]) == 1


class TestGenerateCoachText:
    """Tests for generate_coach_text method."""

    @patch("requests.post")
    def test_generate_coach_text_uses_openai_api(self, mock_post: MagicMock, runtime):
        """Should use OpenAI-compatible API for coach responses."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Coach response text"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            runtime.refresh_configuration(force=True)
            result = runtime.generate_coach_text(user_message="How should I train?")

            assert result == "Coach response text"


class TestRuntimeInfo:
    """Tests for runtime info discovery."""

    def test_runtime_info_without_api_configured(self, runtime):
        """Should report unavailable when no API configured and no local model."""
        with patch.dict(os.environ, {}, clear=True):
            runtime.refresh_configuration(force=True)
            info = runtime.info()

            # Without API key or local model, should not be available
            assert info.available is False
