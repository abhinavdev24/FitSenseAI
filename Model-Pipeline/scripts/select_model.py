"""Model selection script that compares evaluation results and bias reports.

This script compares N candidate models based on evaluation metrics and bias detection
results, computes a weighted composite score, and selects the best model. Outputs a
detailed selection report with scoring breakdown and rationale.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def load_eval_results(eval_dir: str) -> dict[str, Any]:
    """Load evaluation results from a directory.

    Args:
        eval_dir: Path to directory containing evaluation_results.json

    Returns:
        Parsed evaluation results dict with model_name and metrics

    Raises:
        FileNotFoundError: If evaluation_results.json not found
        json.JSONDecodeError: If JSON is malformed
    """
    eval_path = Path(eval_dir) / "evaluation_results.json"
    try:
        with open(eval_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Evaluation results not found at {eval_path}") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON at {eval_path}", e.doc, e.pos
        ) from e


def load_bias_report(bias_dir: str) -> dict[str, Any]:
    """Load bias detection report from a directory.

    Args:
        bias_dir: Path to directory containing bias_report.json

    Returns:
        Parsed bias report dict with bias_detected, flagged_slices, overall_metrics

    Raises:
        FileNotFoundError: If bias_report.json not found
        json.JSONDecodeError: If JSON is malformed
    """
    bias_path = Path(bias_dir) / "bias_report.json"
    try:
        with open(bias_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Bias report not found at {bias_path}") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse JSON at {bias_path}", e.doc, e.pos
        ) from e


def compute_bias_score(bias_report: dict[str, Any]) -> float:
    """Compute bias score as mean deviation across flagged slices.

    Args:
        bias_report: Bias detection report dict

    Returns:
        Float between 0 and 1, where 0 = no bias, higher = more bias
    """
    flagged_slices = bias_report.get("flagged_slices", [])
    if not flagged_slices:
        return 0.0

    deviations = [slice_info.get("deviation", 0.0) for slice_info in flagged_slices]
    return sum(deviations) / len(deviations) if deviations else 0.0


def normalize_val_loss(losses: list[float]) -> list[float]:
    """Min-max normalize validation losses for N models.

    Lower loss is better. Returns scores in [0, 1] where higher = better
    (i.e., the model with lowest loss gets score 1.0).

    Args:
        losses: List of N val_loss values

    Returns:
        List of N normalized scores (0-1, higher = better)
    """
    if not losses:
        raise ValueError("losses must be non-empty")

    if len(losses) == 1:
        return [1.0]

    finite = [loss for loss in losses if loss != float("inf")]
    if not finite:
        return [0.0] * len(losses)

    min_loss = min(finite)
    max_loss = max(finite)
    denominator = max_loss - min_loss + 1e-8

    # Invert: lower loss -> higher normalized score
    return [
        0.0 if loss == float("inf") else 1.0 - (loss - min_loss) / denominator
        for loss in losses
    ]


def compute_composite_score(
    eval_results: dict[str, Any],
    bias_score: float,
    normalized_loss: float,
) -> tuple[float, dict[str, float]]:
    """Compute weighted composite score for a model.

    Args:
        eval_results: Evaluation results dict with metrics
        bias_score: Bias score (0 = no bias, higher = more bias)
        normalized_loss: Normalized val_loss (0-1 range, higher = better)

    Returns:
        Tuple of (composite_score, breakdown_dict) where breakdown shows per-metric contributions
    """
    metrics = eval_results.get("metrics", {})

    # Extract metrics with defaults
    tool_call_accuracy = metrics.get("tool_call_accuracy", 0.0)
    json_parse_rate = metrics.get("json_parse_rate", 0.0)
    schema_compliance = metrics.get("schema_compliance", 0.0)
    thinking_presence_rate = metrics.get("thinking_presence_rate", 0.0)

    # Invert bias score (higher bias -> lower contribution)
    bias_contribution = 1.0 - bias_score

    # Weights
    weights = {
        "tool_call_accuracy": 0.30,
        "json_parse_rate": 0.20,
        "schema_compliance": 0.15,
        "thinking_presence_rate": 0.10,
        "val_loss": 0.15,
        "bias_score": 0.10,
    }

    # Compute breakdown
    breakdown = {
        "tool_call_accuracy": tool_call_accuracy * weights["tool_call_accuracy"],
        "json_parse_rate": json_parse_rate * weights["json_parse_rate"],
        "schema_compliance": schema_compliance * weights["schema_compliance"],
        "thinking_presence_rate": thinking_presence_rate * weights["thinking_presence_rate"],
        "val_loss": normalized_loss * weights["val_loss"],
        "bias_score": bias_contribution * weights["bias_score"],
    }

    composite = sum(breakdown.values())
    return composite, breakdown


def generate_rationale(
    scores: dict[str, dict[str, Any]],
    eval_results_map: dict[str, dict[str, Any]],
) -> str:
    """Generate human-readable rationale for model selection.

    Args:
        scores: Dict mapping candidate IDs to score dicts
        eval_results_map: Dict mapping candidate IDs to evaluation results

    Returns:
        Human-readable rationale string
    """
    ranked = sorted(scores.keys(), key=lambda m: scores[m]["composite_score"], reverse=True)
    selected = ranked[0]
    selected_score = scores[selected]["composite_score"]

    if len(ranked) > 1:
        runner_up = ranked[1]
        runner_up_score = scores[runner_up]["composite_score"]
        score_diff = selected_score - runner_up_score

        selected_metrics = eval_results_map[selected].get("metrics", {})
        runner_up_metrics = eval_results_map[runner_up].get("metrics", {})

        metric_diffs = {}
        for key in ["tool_call_accuracy", "json_parse_rate", "schema_compliance"]:
            metric_diffs[key] = selected_metrics.get(key, 0.0) - runner_up_metrics.get(key, 0.0)

        primary_metric = max(metric_diffs.keys(), key=lambda k: metric_diffs[k])
        selected_val = selected_metrics.get(primary_metric, 0.0)
        runner_up_val = runner_up_metrics.get(primary_metric, 0.0)

        return (
            f"{selected} selected with composite score {selected_score:.4f} "
            f"(runner-up: {runner_up} at {runner_up_score:.4f}, margin: {score_diff:.4f}). "
            f"Primary differentiator: {primary_metric} ({selected_val:.4f} vs {runner_up_val:.4f})."
        )
    else:
        return f"{selected} selected (only candidate) with composite score {selected_score:.4f}."


def log_comparison_table(
    scores: dict[str, dict[str, Any]],
    eval_results_map: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Log a formatted comparison table and return it as structured data.

    Args:
        scores: Dict mapping candidate IDs to score dicts
        eval_results_map: Dict mapping candidate IDs to evaluation results
        logger: Logger instance

    Returns:
        List of dicts for JSON serialization
    """
    models = sorted(scores.keys())

    logger.info("=" * 100)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * 100)

    # Build table rows
    table_rows = []
    for model in models:
        metrics = eval_results_map[model].get("metrics", {})
        row = {
            "model": model,
            "tool_call_accuracy": metrics.get("tool_call_accuracy", 0.0),
            "json_parse_rate": metrics.get("json_parse_rate", 0.0),
            "schema_compliance": metrics.get("schema_compliance", 0.0),
            "thinking_presence_rate": metrics.get("thinking_presence_rate", 0.0),
            "val_loss": metrics.get("val_loss", 0.0),
            "composite_score": scores[model]["composite_score"],
        }
        table_rows.append(row)

    # Log header
    header = "{:<40} {:<18} {:<16} {:<18} {:<22} {:<12} {:<18}".format(
        "Model",
        "Tool Accuracy",
        "JSON Parse",
        "Schema Comp.",
        "Thinking Presence",
        "Val Loss",
        "Composite Score",
    )
    logger.info(header)
    logger.info("-" * 100)

    # Log rows
    for row in table_rows:
        line = "{:<40} {:<18.4f} {:<16.4f} {:<18.4f} {:<22.4f} {:<12.4f} {:<18.4f}".format(
            row["model"],
            row["tool_call_accuracy"],
            row["json_parse_rate"],
            row["schema_compliance"],
            row["thinking_presence_rate"],
            row["val_loss"],
            row["composite_score"],
        )
        logger.info(line)

    logger.info("=" * 100)
    return table_rows


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results and bias reports to select best model."
    )
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        required=True,
        metavar="EVAL_DIR",
        help="One or more evaluation output directories (each contains evaluation_results.json)",
    )
    parser.add_argument(
        "--bias-dirs",
        nargs="+",
        required=True,
        metavar="BIAS_DIR",
        help="One or more bias detection output directories (each contains bias_report.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="Model-Pipeline/outputs/selection",
        help="Output directory for selected_model.json (default: Model-Pipeline/outputs/selection)",
    )
    parser.add_argument(
        "--require-no-bias",
        action="store_true",
        help="Flag: if set, reject any model with bias_detected=true and require manual review",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting model selection process")

        # Validate that eval-dirs and bias-dirs have the same count
        if len(args.eval_dirs) != len(args.bias_dirs):
            raise ValueError(
                f"--eval-dirs ({len(args.eval_dirs)}) and --bias-dirs ({len(args.bias_dirs)}) "
                f"must have the same number of entries"
            )

        # Load all N candidates using directory basename as identifier
        eval_results_map: dict[str, dict] = {}
        bias_reports_map: dict[str, dict] = {}

        for eval_dir, bias_dir in zip(args.eval_dirs, args.bias_dirs):
            candidate_id = Path(eval_dir).name
            eval_results_map[candidate_id] = load_eval_results(eval_dir)
            bias_reports_map[candidate_id] = load_bias_report(bias_dir)

        logger.info(
            f"Loaded {len(eval_results_map)} candidates: {list(eval_results_map.keys())}"
        )

        # Compute bias scores
        bias_scores = {cid: compute_bias_score(br) for cid, br in bias_reports_map.items()}
        logger.info(
            "Bias scores: "
            + ", ".join(f"{cid}={score:.4f}" for cid, score in bias_scores.items())
        )

        # Normalize val losses
        val_losses = [
            eval_results_map[cid].get("metrics", {}).get("val_loss", float("inf"))
            for cid in eval_results_map
        ]
        norm_losses = normalize_val_loss(val_losses)
        norm_loss_map = dict(zip(eval_results_map.keys(), norm_losses))

        # Compute composite scores
        scores: dict[str, dict] = {}
        for cid in eval_results_map:
            composite, breakdown = compute_composite_score(
                eval_results_map[cid], bias_scores[cid], norm_loss_map[cid]
            )
            scores[cid] = {"composite_score": composite, "breakdown": breakdown}

        logger.info(
            "Composite scores: "
            + ", ".join(f"{cid}={s['composite_score']:.4f}" for cid, s in scores.items())
        )

        # Check for bias if required
        decision = "auto"
        if args.require_no_bias:
            biased = [
                cid for cid, br in bias_reports_map.items() if br.get("bias_detected", False)
            ]
            if biased:
                logger.warning(
                    f"Bias detected in {len(biased)} candidate(s): {biased}. "
                    f"Manual review required."
                )
                decision = "manual_review_required"

        # Generate rationale
        rationale = generate_rationale(scores, eval_results_map)
        logger.info(f"Rationale: {rationale}")

        # Log comparison table
        comparison_table = log_comparison_table(scores, eval_results_map, logger)

        # Build bias status
        bias_status = {
            cid: {
                "bias_detected": bias_reports_map[cid].get("bias_detected", False),
                "n_flagged": len(bias_reports_map[cid].get("flagged_slices", [])),
            }
            for cid in bias_reports_map
        }

        # Determine selected model
        if decision == "auto":
            selected_model = max(scores.keys(), key=lambda m: scores[m]["composite_score"])
        else:
            selected_model = None

        # Create output
        output = {
            "selected_model": selected_model,
            "decision": decision,
            "scores": scores,
            "comparison_table": comparison_table,
            "rationale": rationale,
            "bias_status": bias_status,
            "timestamp": datetime.now().isoformat(),
        }

        # Write output file
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "selected_model.json"

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Selection results written to {output_path}")
        logger.info(f"Selected model: {selected_model}")

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error during model selection: {e}")
        raise


if __name__ == "__main__":
    main()
