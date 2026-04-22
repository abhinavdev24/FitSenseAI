"""
Unit tests for Model-Deployment/scripts/monitor_deployment.py

Tests cover all pure-logic functions. No real HTTP calls are made.
"""

import json
import sys
from pathlib import Path

# Make the scripts directory importable without installing it as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import monitor_deployment as mon  # noqa: E402


# =============================================================================
# strip_think
# =============================================================================

class TestStripThink:
    def test_removes_think_block(self) -> None:
        text = "<think>some reasoning</think>result"
        assert mon.strip_think(text) == "result"

    def test_no_think_block_unchanged(self) -> None:
        text = '{"plan_name": "test", "days": []}'
        assert mon.strip_think(text) == text

    def test_multiline_think_removed(self) -> None:
        text = "<think>\nline one\nline two\n</think>\n{}"
        assert mon.strip_think(text) == "{}"

    def test_empty_string(self) -> None:
        assert mon.strip_think("") == ""

    def test_think_block_with_surrounding_whitespace(self) -> None:
        text = "  <think>x</think>  result  "
        assert mon.strip_think(text) == "result"


# =============================================================================
# is_valid_json
# =============================================================================

class TestIsValidJson:
    def test_plain_json_object(self) -> None:
        assert mon.is_valid_json('{"key": "value"}') is True

    def test_plain_json_array(self) -> None:
        assert mon.is_valid_json('[1, 2, 3]') is True

    def test_invalid_json(self) -> None:
        assert mon.is_valid_json("not json at all") is False

    def test_empty_string_is_invalid(self) -> None:
        assert mon.is_valid_json("") is False

    def test_json_after_think_block(self) -> None:
        text = '<think>reasoning here</think>{"plan_name": "test", "days": []}'
        assert mon.is_valid_json(text) is True

    def test_partial_json_is_invalid(self) -> None:
        assert mon.is_valid_json('{"key": ') is False


# =============================================================================
# has_schema
# =============================================================================

class TestHasSchema:
    def test_valid_schema(self) -> None:
        text = '{"plan_name": "Push Day", "days": [{"name": "DAY_1"}]}'
        assert mon.has_schema(text) is True

    def test_missing_days_key(self) -> None:
        assert mon.has_schema('{"plan_name": "Push Day"}') is False

    def test_missing_plan_name_key(self) -> None:
        assert mon.has_schema('{"days": []}') is False

    def test_both_keys_required(self) -> None:
        assert mon.has_schema('{"other": "value"}') is False

    def test_schema_after_think_block(self) -> None:
        text = '<think>ok</think>{"plan_name": "A", "days": []}'
        assert mon.has_schema(text) is True

    def test_json_array_not_schema_compliant(self) -> None:
        assert mon.has_schema('[1, 2, 3]') is False

    def test_invalid_json_returns_false(self) -> None:
        assert mon.has_schema("not json") is False


# =============================================================================
# has_thinking
# =============================================================================

class TestHasThinking:
    def test_detects_think_block(self) -> None:
        assert mon.has_thinking("<think>some reasoning</think>result") is True

    def test_no_think_block(self) -> None:
        assert mon.has_thinking('{"plan_name": "test"}') is False

    def test_multiline_think_detected(self) -> None:
        assert mon.has_thinking("<think>\nlong\nreasoning\n</think>{}") is True

    def test_empty_string(self) -> None:
        assert mon.has_thinking("") is False


# =============================================================================
# check_thresholds
# =============================================================================

class TestCheckThresholds:
    def test_all_thresholds_pass(self) -> None:
        metrics = {
            "json_validity_rate": 0.90,
            "schema_compliance": 0.85,
            "avg_latency_ms": 3000.0,
        }
        passed, violations = mon.check_thresholds(metrics, mon.DEFAULT_THRESHOLDS, None)
        assert passed is True
        assert violations == []

    def test_json_validity_below_min_triggers_retrain(self) -> None:
        metrics = {
            "json_validity_rate": 0.50,   # below 0.70
            "schema_compliance": 0.80,
            "avg_latency_ms": 3000.0,
        }
        passed, violations = mon.check_thresholds(metrics, mon.DEFAULT_THRESHOLDS, None)
        assert passed is False
        assert any("json_validity_rate" in v and "[RETRAIN]" in v for v in violations)

    def test_schema_compliance_below_min_triggers_retrain(self) -> None:
        metrics = {
            "json_validity_rate": 0.90,
            "schema_compliance": 0.40,    # below 0.60
            "avg_latency_ms": 3000.0,
        }
        passed, violations = mon.check_thresholds(metrics, mon.DEFAULT_THRESHOLDS, None)
        assert passed is False
        assert any("schema_compliance" in v and "[RETRAIN]" in v for v in violations)

    def test_high_latency_is_alert_only_not_retrain(self) -> None:
        # Latency breach creates an [ALERT] violation but must NOT trigger retrain
        metrics = {
            "json_validity_rate": 0.90,
            "schema_compliance": 0.85,
            "avg_latency_ms": 9000.0,     # above 8000
        }
        passed, violations = mon.check_thresholds(metrics, mon.DEFAULT_THRESHOLDS, None)
        assert passed is True
        assert any("avg_latency_ms" in v and "[ALERT]" in v for v in violations)

    def test_regression_above_10pct_triggers_retrain(self) -> None:
        # json_validity_rate: 0.72 is above min (0.70) but 20% below baseline (0.90)
        metrics = {
            "json_validity_rate": 0.72,
            "schema_compliance": 0.65,
            "avg_latency_ms": 3000.0,
        }
        baseline = {
            "metrics": {
                "json_validity_rate": 0.90,  # (0.90 - 0.72) / 0.90 = 20% regression
                "schema_compliance": 0.70,   # (0.70 - 0.65) / 0.70 = 7.1% → under 10%
            }
        }
        passed, violations = mon.check_thresholds(metrics, mon.DEFAULT_THRESHOLDS, baseline)
        assert passed is False
        assert any("regressed" in v for v in violations)

    def test_regression_under_10pct_passes(self) -> None:
        # 4.4% regression — within 10% tolerance
        metrics = {
            "json_validity_rate": 0.86,
            "schema_compliance": 0.65,
            "avg_latency_ms": 3000.0,
        }
        baseline = {
            "metrics": {
                "json_validity_rate": 0.90,  # (0.90 - 0.86) / 0.90 = 4.4%
                "schema_compliance": 0.70,   # (0.70 - 0.65) / 0.70 = 7.1%
            }
        }
        passed, violations = mon.check_thresholds(metrics, mon.DEFAULT_THRESHOLDS, baseline)
        assert passed is True

    def test_none_metric_value_is_skipped(self) -> None:
        # avg_latency_ms missing from metrics — should not raise
        metrics = {
            "json_validity_rate": 0.90,
            "schema_compliance": 0.85,
        }
        passed, violations = mon.check_thresholds(metrics, mon.DEFAULT_THRESHOLDS, None)
        assert passed is True


# =============================================================================
# load_prompts
# =============================================================================

class TestLoadPrompts:
    def test_falls_back_to_builtins_when_file_missing(self, tmp_path: Path) -> None:
        prompts = mon.load_prompts(tmp_path / "nonexistent.jsonl", n_samples=3)
        assert len(prompts) >= 1
        assert all("system" in p and "user" in p for p in prompts)

    def test_loads_from_valid_jsonl(self, tmp_path: Path) -> None:
        val_path = tmp_path / "val.jsonl"
        records = [
            {
                "messages": [
                    {"role": "system", "content": f"system {i}"},
                    {"role": "user", "content": f"user prompt {i}"},
                    {"role": "assistant", "content": f"response {i}"},
                ]
            }
            for i in range(5)
        ]
        val_path.write_text("\n".join(json.dumps(r) for r in records))

        prompts = mon.load_prompts(val_path, n_samples=3)
        assert len(prompts) == 3
        assert prompts[0]["system"] == "system 0"
        assert prompts[0]["user"] == "user prompt 0"

    def test_respects_n_samples_limit(self, tmp_path: Path) -> None:
        val_path = tmp_path / "val.jsonl"
        records = [
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": f"u{i}"},
                    {"role": "assistant", "content": "a"},
                ]
            }
            for i in range(10)
        ]
        val_path.write_text("\n".join(json.dumps(r) for r in records))

        prompts = mon.load_prompts(val_path, n_samples=4)
        assert len(prompts) == 4

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        val_path = tmp_path / "val.jsonl"
        lines = [
            "not valid json",
            json.dumps({"messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "valid prompt"},
                {"role": "assistant", "content": "a"},
            ]}),
            "",
        ]
        val_path.write_text("\n".join(lines))

        prompts = mon.load_prompts(val_path, n_samples=5)
        assert len(prompts) == 1
        assert prompts[0]["user"] == "valid prompt"

    def test_falls_back_to_builtins_when_file_has_no_valid_prompts(
        self, tmp_path: Path
    ) -> None:
        val_path = tmp_path / "val.jsonl"
        val_path.write_text("bad line\nanother bad line\n")

        prompts = mon.load_prompts(val_path, n_samples=3)
        assert len(prompts) >= 1  # got built-ins


# =============================================================================
# detect_drift
# =============================================================================

class TestDetectDrift:
    def _write_val(self, path: Path, user_len: int, n: int = 5) -> None:
        records = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "x" * user_len},
                    {"role": "assistant", "content": "resp"},
                ]
            }
            for _ in range(n)
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))

    def test_skipped_when_val_missing(self, tmp_path: Path) -> None:
        result = mon.detect_drift([{"user": "hi"}], tmp_path / "none.jsonl")
        assert result["status"] == "skipped"

    def test_no_drift_within_threshold(self, tmp_path: Path) -> None:
        val_path = tmp_path / "val.jsonl"
        self._write_val(val_path, user_len=100)

        # Current prompts ~5% shorter — well within 25% threshold
        prompts = [{"system": "s", "user": "x" * 96}]
        result = mon.detect_drift(prompts, val_path)
        assert result["flagged"] is False
        assert result["status"] == "ok"

    def test_drift_flagged_when_over_threshold(self, tmp_path: Path) -> None:
        val_path = tmp_path / "val.jsonl"
        self._write_val(val_path, user_len=100)

        # Current prompts are 90% shorter — well over 25% threshold
        prompts = [{"system": "s", "user": "x" * 10}]
        result = mon.detect_drift(prompts, val_path)
        assert result["flagged"] is True
        assert result["relative_shift"] > 0.25
        assert result["status"] == "flagged"
