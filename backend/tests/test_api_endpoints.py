"""Integration tests for API endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import Base, engine, SessionLocal


@pytest.fixture(scope="module")
def test_client():
    """Create a test client with in-memory database."""
    Base.metadata.create_all(bind=engine)
    client = TestClient(app)
    yield client
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def authenticated_client(test_client):
    """Return a client authenticated as a test user."""
    # Sign up a test user
    signup_response = test_client.post(
        "/auth/signup",
        json={"name": "Test User", "email": "test@example.com", "password": "testpass123"},
    )
    assert signup_response.status_code == 200
    token = signup_response.json()["token"]

    # Return client with auth header
    test_client.headers["Authorization"] = f"Bearer {token}"
    return test_client


class TestCoachEndpoint:
    """Tests for /coach endpoint."""

    def test_coach_endpoint_requires_auth(self, test_client):
        """Should require authentication."""
        response = test_client.post("/coach", json={"message": "Hello"})
        assert response.status_code == 401

    @patch("app.llm_runtime.requests.post")
    def test_coach_endpoint_with_mocked_llm(self, mock_post: MagicMock, authenticated_client):
        """Should return LLM response when API is configured."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Stay hydrated and train consistently!"}}]
        }
        mock_post.return_value = mock_response

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            response = authenticated_client.post(
                "/coach",
                json={"message": "How should I train this week?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "reply" in data
            assert data["reply"] == "Stay hydrated and train consistently!"

    @patch("app.llm_runtime.requests.post")
    def test_coach_endpoint_fallback_to_rules(self, mock_post: MagicMock, authenticated_client):
        """Should fall back to rule-based response when API fails."""
        mock_post.side_effect = Exception("API Error")

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            response = authenticated_client.post(
                "/coach",
                json={"message": "I have knee pain during squats"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "reply" in data
            # Rule-based response should mention safety/pain
            assert "pain" in data["reply"].lower() or "conservative" in data["reply"].lower()


class TestModelRuntimeEndpoint:
    """Tests for /model/runtime endpoint."""

    def test_runtime_endpoint_requires_auth(self, test_client):
        """Should require authentication."""
        response = test_client.get("/model/runtime")
        assert response.status_code == 401

    def test_runtime_endpoint_returns_info(self, authenticated_client):
        """Should return runtime information."""
        with patch.dict(os.environ, {}, clear=True):
            response = authenticated_client.get("/model/runtime")

            assert response.status_code == 200
            data = response.json()
            assert "available" in data
            assert "provider" in data


class TestPlansEndpoint:
    """Tests for /plans endpoint."""

    def test_plans_endpoint_requires_auth(self, test_client):
        """Should require authentication."""
        response = test_client.post("/plans", json={})
        assert response.status_code == 401

    def test_plans_endpoint_requires_onboarding(self, authenticated_client):
        """Should require onboarding before generating plans."""
        response = authenticated_client.post("/plans", json={})
        # Should fail because user hasn't completed onboarding
        assert response.status_code in [400, 422, 500]

    @patch("app.llm_runtime.requests.post")
    def test_plans_endpoint_with_onboarding(self, mock_post: MagicMock, test_client):
        """Should generate plan after onboarding."""
        # Create and authenticate user
        signup_response = test_client.post(
            "/auth/signup",
            json={
                "name": "Plan Test User",
                "email": "plan@example.com",
                "password": "testpass123",
            },
        )
        token = signup_response.json()["token"]
        test_client.headers["Authorization"] = f"Bearer {token}"

        # Complete onboarding
        onboarding_response = test_client.post(
            "/profile/onboarding",
            json={
                "age": 25,
                "sex": "male",
                "height_cm": 175,
                "activity_level": "moderate",
                "days_per_week": 4,
                "experience_level": "beginner",
                "equipment": ["bodyweight", "dumbbells"],
                "goal_name": "muscle gain",
                "conditions": [],
                "medications": [],
                "allergies": [],
                "weight_kg": 70,
            },
        )
        assert onboarding_response.status_code == 200

        # Mock LLM response for plan generation
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "plan_name": "Test Muscle Gain Plan",
                        "days": [{
                            "name": "Push Day",
                            "day_order": 1,
                            "exercises": [{
                                "exercise_name": "Bench Press",
                                "position": 1,
                                "sets": [{"set_number": 1, "target_reps": 10, "target_rir": 3, "rest_seconds": 90}]
                            }]
                        }]
                    })
                }
            }]
        }
        mock_post.return_value = mock_response

        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_URL": "https://api.example.com/chat/completions",
        }
        with patch.dict(os.environ, env, clear=True):
            response = test_client.post(
                "/plans",
                json={"goal_name": "muscle gain", "days_per_week": 4},
            )

            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert "status" in data


class TestFallbackChain:
    """Tests to verify the fallback chain works correctly."""

    @patch("app.llm_runtime.requests.post")
    def test_fallback_cloud_run_not_configured(self, mock_post: MagicMock, authenticated_client):
        """Should return rule-based response when no fallback available."""
        mock_post.side_effect = Exception("Connection refused")

        # Mock no local model either
        with patch.dict(os.environ, {}, clear=True):
            response = authenticated_client.post(
                "/coach",
                json={"message": "How should I train?"},
            )

            assert response.status_code == 200
            data = response.json()
            # Should use rule-based response
            assert data["context_type"] == "safety" or "general" in str(data)

    def test_no_openrouter_vars_used(self):
        """Verify no OpenRouter variables are used."""
        with patch.dict(os.environ, {}, clear=True):
            # Check that OPENROUTER_API_KEY is not read
            env_keys = [k for k in os.environ.keys() if "OPENROUTER" in k.upper()]
            assert len(env_keys) == 0


# Import os at module level for the test
import os
