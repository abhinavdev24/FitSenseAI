"""Tests for Model-Pipeline/scripts/select_model.py.

All functions under test are pure (no I/O, no GPU) — they can run anywhere.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_results(
    tool_call_accuracy: float = 0.8,
    json_parse_rate: float = 0.9,
    schema_compliance: float = 0.85,
    thinking_presence_rate: float = 0.95,
    val_loss: float = 0.36,
) -> dict[str, Any]:
    return {
        "model_name": "unsloth/Qwen3-4B",
        "metrics": {
            "tool_call_accuracy": tool_call_accuracy,
            "json_parse_rate": json_parse_rate,
            "schema_compliance": schema_compliance,
            "thinking_presence_rate": thinking_presence_rate,
            "val_loss": val_loss,
        },
    }


def _bias_report(flagged: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "bias_detected": bool(flagged),
        "flagged_slices": flagged or [],
    }


# ---------------------------------------------------------------------------
# Tests for compute_bias_score
# ---------------------------------------------------------------------------

class TestComputeBiasScore:
    def test_no_flagged_slices_returns_zero(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        assert mod.compute_bias_score(_bias_report()) == 0.0

    def test_single_slice_returns_its_deviation(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        report = _bias_report([{"deviation": 0.5, "dimension": "age_group"}])
        assert mod.compute_bias_score(report) == 0.5

    def test_multiple_slices_returns_mean(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        report = _bias_report([
            {"deviation": 0.2},
            {"deviation": 0.4},
            {"deviation": 0.6},
        ])
        score = mod.compute_bias_score(report)
        assert abs(score - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# Tests for normalize_val_loss
# ---------------------------------------------------------------------------

class TestNormalizeValLoss:
    def test_single_loss_returns_one(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        result = mod.normalize_val_loss([0.4])
        assert result == [1.0]

    def test_lower_loss_gets_higher_score(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        # loss 0.3 < 0.5, so index 0 should have a higher score
        result = mod.normalize_val_loss([0.3, 0.5])
        assert result[0] > result[1]

    def test_lowest_loss_gets_score_one(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        result = mod.normalize_val_loss([0.3, 0.5, 0.7])
        assert abs(result[0] - 1.0) < 1e-6  # 0.3 is lowest → score 1.0

    def test_highest_loss_gets_score_zero(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        result = mod.normalize_val_loss([0.3, 0.5, 0.7])
        assert abs(result[2] - 0.0) < 1e-6  # 0.7 is highest → score 0.0

    def test_all_inf_returns_zeros(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        result = mod.normalize_val_loss([float("inf"), float("inf")])
        assert result == [0.0, 0.0]

    def test_mixed_inf_gets_zero(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        result = mod.normalize_val_loss([0.4, float("inf")])
        assert result[0] == 1.0   # finite → best
        assert result[1] == 0.0   # inf → worst


# ---------------------------------------------------------------------------
# Tests for compute_composite_score
# ---------------------------------------------------------------------------

class TestComputeCompositeScore:
    def test_perfect_model_high_score(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        perfect = _eval_results(
            tool_call_accuracy=1.0,
            json_parse_rate=1.0,
            schema_compliance=1.0,
            thinking_presence_rate=1.0,
        )
        score, _ = mod.compute_composite_score(perfect, bias_score=0.0, normalized_loss=1.0)
        # All weights sum to 1.0, all contributions at max → score should be 1.0
        assert abs(score - 1.0) < 1e-9

    def test_zero_model_low_score(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        zero = _eval_results(
            tool_call_accuracy=0.0,
            json_parse_rate=0.0,
            schema_compliance=0.0,
            thinking_presence_rate=0.0,
        )
        score, _ = mod.compute_composite_score(zero, bias_score=1.0, normalized_loss=0.0)
        assert score == 0.0

    def test_spot_check_weighted_sum(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        results = _eval_results(
            tool_call_accuracy=0.8,   # weight 0.30 → 0.24
            json_parse_rate=0.9,      # weight 0.20 → 0.18
            schema_compliance=0.7,   # weight 0.15 → 0.105
            thinking_presence_rate=1.0,  # weight 0.10 → 0.10
        )
        # normalized_loss=0.5, weight 0.15 → 0.075
        # bias_score=0.2 → bias_contribution=0.8, weight 0.10 → 0.08
        expected = 0.24 + 0.18 + 0.105 + 0.10 + 0.075 + 0.08
        score, breakdown = mod.compute_composite_score(
            results, bias_score=0.2, normalized_loss=0.5
        )
        assert abs(score - expected) < 1e-9
        assert set(breakdown.keys()) == {
            "tool_call_accuracy", "json_parse_rate", "schema_compliance",
            "thinking_presence_rate", "val_loss", "bias_score",
        }

    def test_breakdown_sums_to_composite(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        score, breakdown = mod.compute_composite_score(
            _eval_results(), bias_score=0.1, normalized_loss=0.8
        )
        assert abs(sum(breakdown.values()) - score) < 1e-9


# ---------------------------------------------------------------------------
# Tests for generate_rationale
# ---------------------------------------------------------------------------

class TestGenerateRationale:
    def _make_scores(self, score_map: dict[str, float]) -> dict[str, dict[str, Any]]:
        return {name: {"composite_score": s} for name, s in score_map.items()}

    def test_single_candidate_says_only_candidate(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        scores = self._make_scores({"final_adapter": 0.75})
        eval_map = {"final_adapter": _eval_results()}
        rationale = mod.generate_rationale(scores, eval_map)
        assert "only candidate" in rationale
        assert "final_adapter" in rationale

    def test_two_candidates_winner_is_named(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        scores = self._make_scores({"checkpoint-260": 0.82, "checkpoint-250": 0.76})
        eval_map = {
            "checkpoint-260": _eval_results(tool_call_accuracy=0.9),
            "checkpoint-250": _eval_results(tool_call_accuracy=0.7),
        }
        rationale = mod.generate_rationale(scores, eval_map)
        assert "checkpoint-260" in rationale  # winner
        assert "checkpoint-250" in rationale  # runner-up

    def test_highest_composite_score_wins(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        scores = self._make_scores({
            "ckpt-a": 0.60,
            "ckpt-b": 0.85,
            "ckpt-c": 0.72,
        })
        eval_map = {k: _eval_results() for k in scores}
        rationale = mod.generate_rationale(scores, eval_map)
        assert "ckpt-b" in rationale  # highest score

    def test_rationale_mentions_runner_up(self, module_loader: Any) -> None:
        mod = module_loader("scripts/select_model.py")
        scores = self._make_scores({"best": 0.9, "second": 0.7, "third": 0.5})
        eval_map = {k: _eval_results() for k in scores}
        rationale = mod.generate_rationale(scores, eval_map)
        assert "runner-up" in rationale or "second" in rationale
