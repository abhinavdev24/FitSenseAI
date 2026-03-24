"""
Bias detection script for the FitSenseAI fine-tuned QLoRA adapter.

Slices the validation set by demographic and contextual attributes extracted
from user message content (age group, gender, fitness level, goal type, BMI
category), computes per-slice metrics, flags slices that deviate beyond a
configurable threshold from the overall mean, and writes a bias_report.json
plus heatmap visualisations.

Usage:
    python bias_detection.py \\
        --adapter-dir Model-Pipeline/outputs/final_adapter \\
        --config Model-Pipeline/config/training_config.yaml \\
        --output-dir Model-Pipeline/outputs/bias_detection \\
        --threshold 0.1 \\
        --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Ensure sibling modules are importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Self-contained helpers (reproduced here to avoid triggering module-level
# heavy imports in train.py / evaluate.py at import time)
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to stdout with a standard format.

    Args:
        name: Logger name.
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_config(config_path: str) -> dict[str, Any]:
    """Load training config from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def get_git_commit() -> str | None:
    """Return the current HEAD commit SHA, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Slice-attribute extraction
# ---------------------------------------------------------------------------

_AGE_PATTERN = re.compile(
    r"(?:age[:\s]+|(\b))(\d{1,3})\s*(?:years?\s*old|yr)",
    re.IGNORECASE,
)
_AGE_COLON_PATTERN = re.compile(r"age\s*:\s*(\d{1,3})", re.IGNORECASE)
_BMI_PATTERN = re.compile(r"bmi\s*(?:of|:)?\s*(\d{1,2}(?:\.\d+)?)", re.IGNORECASE)

_GENDER_PATTERNS: dict[str, re.Pattern[str]] = {
    "non-binary": re.compile(
        r"non[-\s]binary|nonbinary|they/them", re.IGNORECASE
    ),
    "female": re.compile(
        r"\bfemale\b|\bwoman\b|\bwomen\b|\bshe\b|\bher\b", re.IGNORECASE
    ),
    "male": re.compile(
        r"\bmale\b|\bman\b|\bmen\b|\bhe\b|\bhis\b", re.IGNORECASE
    ),
}

_FITNESS_KEYWORDS: list[str] = ["beginner", "intermediate", "advanced"]

_GOAL_KEYWORDS: dict[str, list[str]] = {
    "weight_loss": ["weight loss", "lose weight", "fat loss", "cutting"],
    "muscle_gain": ["muscle gain", "build muscle", "bulking", "hypertrophy"],
    "endurance": ["endurance", "cardio", "stamina", "running", "cycling"],
    "flexibility": ["flexibility", "stretching", "yoga", "mobility"],
    "general_fitness": ["general fitness", "overall fitness", "stay fit", "stay active", "wellness"],
}


def extract_age_group(text: str) -> str:
    """Extract age from text and bucket into an age group string.

    Looks for patterns like "25 years old", "age: 35", or "age 40".

    Args:
        text: User message content to search.

    Returns:
        One of "18-25", "26-35", "36-45", "46-55", "56+", or "unknown".
    """
    age: int | None = None

    # Try "age: 35" style
    m = _AGE_COLON_PATTERN.search(text)
    if m:
        try:
            age = int(m.group(1))
        except ValueError:
            pass

    # Try "25 years old" / "25 yr" style
    if age is None:
        m = _AGE_PATTERN.search(text)
        if m:
            try:
                age = int(m.group(2))
            except (ValueError, TypeError):
                pass

    if age is None or age < 10 or age > 110:
        return "unknown"

    if age <= 25:
        return "18-25"
    if age <= 35:
        return "26-35"
    if age <= 45:
        return "36-45"
    if age <= 55:
        return "46-55"
    return "56+"


def extract_gender(text: str) -> str:
    """Extract gender from text via regex word-boundary matching.

    Non-binary is checked first to avoid it being shadowed by "male"/"female"
    substrings.

    Args:
        text: User message content to search.

    Returns:
        One of "male", "female", "non-binary", or "unknown".
    """
    # Ordered: non-binary first, then female, then male
    for label, pattern in _GENDER_PATTERNS.items():
        if pattern.search(text):
            return label
    return "unknown"


def extract_fitness_level(text: str) -> str:
    """Extract fitness level from text via keyword matching.

    Args:
        text: User message content to search.

    Returns:
        One of "beginner", "intermediate", "advanced", or "unknown".
    """
    lower = text.lower()
    for level in _FITNESS_KEYWORDS:
        if level in lower:
            return level
    return "unknown"


def extract_goal_type(text: str) -> str:
    """Extract primary fitness goal from text via keyword matching.

    Args:
        text: User message content to search.

    Returns:
        One of "weight_loss", "muscle_gain", "endurance", "flexibility",
        "general_fitness", or "unknown".
    """
    lower = text.lower()
    for goal, keywords in _GOAL_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return goal
    return "unknown"


def extract_bmi_category(text: str) -> str:
    """Extract BMI value from text and return the WHO category string.

    Looks for patterns like "BMI: 25.3" or "BMI of 30".

    Args:
        text: User message content to search.

    Returns:
        One of "underweight", "normal", "overweight", "obese", or "unknown".
    """
    m = _BMI_PATTERN.search(text)
    if not m:
        return "unknown"
    try:
        bmi = float(m.group(1))
    except ValueError:
        return "unknown"

    if bmi < 18.5:
        return "underweight"
    if bmi < 25.0:
        return "normal"
    if bmi < 30.0:
        return "overweight"
    return "obese"


def extract_slice_attributes(user_message: str) -> dict[str, str]:
    """Extract all demographic slice attributes from a user message.

    Args:
        user_message: The raw user message content string.

    Returns:
        Dict mapping dimension name to slice label.  Keys are:
        "age_group", "gender", "fitness_level", "goal_type", "bmi_category".
    """
    return {
        "age_group": extract_age_group(user_message),
        "gender": extract_gender(user_message),
        "fitness_level": extract_fitness_level(user_message),
        "goal_type": extract_goal_type(user_message),
        "bmi_category": extract_bmi_category(user_message),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_bias_evaluation(
    model,
    tokenizer,
    dataset,
    max_samples: int | None,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Generate responses and collect per-sample results with slice attributes.

    For each sample the reference tool_name is extracted from
    ``messages[2]["content"]`` (the ground-truth assistant turn).  The user
    message (``messages[1]["content"]``) is parsed for demographic attributes.

    Args:
        model: Inference-mode model loaded via Unsloth.
        tokenizer: Matching tokenizer.
        dataset: HuggingFace Dataset with a 'messages' column.
        max_samples: If provided, cap the evaluation at this many samples.
        logger: Logger instance.

    Returns:
        List of per-sample result dicts, one per evaluated sample.
    """
    n_samples = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )
    logger.info("Running bias evaluation on %d samples", n_samples)

    # Lazy sibling imports: defer evaluate.py load until inference is needed
    # (evaluate.py transitively imports train.py which imports wandb/unsloth)
    from evaluate import (  # type: ignore[import]
        extract_answer_from_response,
        generate_single,
        parse_tool_name,
    )

    results: list[dict[str, Any]] = []

    for idx in range(n_samples):
        row = dataset[idx]
        messages: list[dict[str, str]] = row["messages"]

        # --- Extract slice attributes from user message ---
        user_content = messages[1]["content"] if len(messages) >= 2 else ""
        slice_attrs = extract_slice_attributes(user_content)

        # --- Extract reference tool name from ground truth ---
        ref_content = messages[2]["content"] if len(messages) >= 3 else ""
        _, ref_json_str = extract_answer_from_response(ref_content)
        ref_tool_name = parse_tool_name(ref_json_str) if ref_json_str else None

        # --- Generate model response ---
        try:
            generated_text, _ = generate_single(model, tokenizer, messages)
        except RuntimeError as exc:
            logger.warning("Sample %d: generation failed — %s", idx, exc)
            record: dict[str, Any] = {
                "sample_idx": idx,
                "ref_tool_name": ref_tool_name,
                "pred_tool_name": None,
                "json_parsed": False,
                "schema_compliant": False,
                "has_thinking": False,
                "response_length": 0,
            }
            record.update(slice_attrs)
            results.append(record)
            continue

        # --- Parse generated output ---
        reasoning, json_str = extract_answer_from_response(generated_text)
        has_thinking = reasoning is not None

        json_parsed = False
        schema_compliant = False
        pred_tool_name: str | None = None

        if json_str:
            try:
                parsed = json.loads(json_str)
                json_parsed = True
                pred_tool_name = parsed.get("tool_name")
                schema_compliant = "tool_name" in parsed and "tool_input" in parsed
            except json.JSONDecodeError:
                pass

        record = {
            "sample_idx": idx,
            "ref_tool_name": ref_tool_name,
            "pred_tool_name": pred_tool_name,
            "json_parsed": json_parsed,
            "schema_compliant": schema_compliant,
            "has_thinking": has_thinking,
            "response_length": len(generated_text),
        }
        record.update(slice_attrs)
        results.append(record)

        if (idx + 1) % 10 == 0:
            logger.info("Progress: %d/%d samples evaluated", idx + 1, n_samples)

    logger.info("Bias evaluation complete — %d samples", len(results))
    return results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

_METRICS = [
    "tool_call_accuracy",
    "json_parse_rate",
    "schema_compliance",
    "thinking_presence_rate",
    "avg_response_length",
]


def _compute_metrics_for_group(
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute the five bias metrics for a list of per-sample result dicts.

    Args:
        samples: Subset of per-sample results belonging to one slice.

    Returns:
        Dict with keys matching _METRICS and a "n_samples" count.
    """
    n = len(samples)
    if n == 0:
        return {m: 0.0 for m in _METRICS} | {"n_samples": 0}

    n_with_ref = sum(1 for s in samples if s["ref_tool_name"] is not None)
    correct_tool = sum(
        1
        for s in samples
        if s["ref_tool_name"] is not None
        and s["pred_tool_name"] == s["ref_tool_name"]
    )
    json_parsed = sum(1 for s in samples if s["json_parsed"])
    schema_compliant = sum(1 for s in samples if s["schema_compliant"])
    has_thinking = sum(1 for s in samples if s["has_thinking"])
    avg_length = sum(s["response_length"] for s in samples) / n

    return {
        "tool_call_accuracy": round(correct_tool / n_with_ref, 6) if n_with_ref > 0 else 0.0,
        "json_parse_rate": round(json_parsed / n, 6),
        "schema_compliance": round(schema_compliant / max(json_parsed, 1), 6),
        "thinking_presence_rate": round(has_thinking / n, 6),
        "avg_response_length": round(avg_length, 2),
        "n_samples": n,
    }


def compute_overall_metrics(
    per_sample_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute aggregate metrics across all samples.

    Args:
        per_sample_results: Full list of per-sample result dicts from
            run_bias_evaluation().

    Returns:
        Dict with the five bias metrics computed over all samples.

    Raises:
        ValueError: If per_sample_results is empty.
    """
    if not per_sample_results:
        raise ValueError("compute_overall_metrics: per_sample_results is empty")
    metrics = _compute_metrics_for_group(per_sample_results)
    # Remove n_samples from the overall dict — it's a top-level field
    metrics.pop("n_samples", None)
    return metrics


def compute_slice_metrics(
    per_sample_results: list[dict[str, Any]],
    dimension: str,
) -> dict[str, dict[str, float]]:
    """Group samples by slice value and compute metrics per group.

    Args:
        per_sample_results: Full list of per-sample result dicts.
        dimension: Slice dimension key, e.g. "age_group" or "gender".

    Returns:
        Dict mapping slice label -> metrics dict (includes "n_samples").
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in per_sample_results:
        slice_val = sample.get(dimension, "unknown")
        groups[slice_val].append(sample)

    return {
        slice_val: _compute_metrics_for_group(samples)
        for slice_val, samples in groups.items()
    }


# ---------------------------------------------------------------------------
# Bias detection
# ---------------------------------------------------------------------------

def detect_bias(
    overall_metrics: dict[str, float],
    sliced_metrics: dict[str, dict[str, dict[str, float]]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Flag slices whose metric deviates beyond the threshold from overall mean.

    A slice is flagged for a metric when:
        abs(slice_value - overall_value) / overall_value > threshold

    Args:
        overall_metrics: Overall metric dict from compute_overall_metrics().
        sliced_metrics: Nested dict: dimension -> slice_label -> metrics.
        threshold: Fractional deviation threshold (default 0.1 = 10 %).

    Returns:
        List of flagged-slice dicts, each with keys: "dimension", "slice",
        "metric", "slice_value", "overall_value", "deviation".
    """
    flagged: list[dict[str, Any]] = []

    for dimension, slices in sliced_metrics.items():
        for slice_label, slice_m in slices.items():
            for metric in _METRICS:
                overall_val = overall_metrics.get(metric, 0.0)
                slice_val = slice_m.get(metric, 0.0)

                # Avoid division by zero for metrics that are 0.0 overall
                if overall_val == 0.0:
                    deviation = 0.0 if slice_val == 0.0 else 1.0
                else:
                    deviation = abs(slice_val - overall_val) / overall_val

                if deviation > threshold:
                    flagged.append(
                        {
                            "dimension": dimension,
                            "slice": slice_label,
                            "metric": metric,
                            "slice_value": round(slice_val, 6),
                            "overall_value": round(overall_val, 6),
                            "deviation": round(deviation, 6),
                        }
                    )

    # Sort by deviation descending so the worst offenders appear first
    flagged.sort(key=lambda x: x["deviation"], reverse=True)
    return flagged


# ---------------------------------------------------------------------------
# Mitigation recommendations
# ---------------------------------------------------------------------------

def generate_mitigation_recommendations(
    flagged_slices: list[dict[str, Any]],
) -> list[str]:
    """Produce human-readable recommendations based on which slices were flagged.

    Args:
        flagged_slices: Output of detect_bias().

    Returns:
        List of recommendation strings.
    """
    if not flagged_slices:
        return ["No significant bias detected. Continue monitoring on future model versions."]

    # Group flagged slices by dimension and metric for concise recommendations
    dim_issues: dict[str, set[str]] = defaultdict(set)
    metric_issues: dict[str, set[str]] = defaultdict(set)

    for entry in flagged_slices:
        dim_issues[entry["dimension"]].add(entry["slice"])
        metric_issues[entry["metric"]].add(entry["dimension"])

    recommendations: list[str] = []

    # Dimension-level recommendations
    dim_advice: dict[str, str] = {
        "age_group": (
            "Collect and up-sample training examples for under-represented age "
            "groups to reduce age-related performance disparity."
        ),
        "gender": (
            "Audit training data for gender-specific language patterns and "
            "balance examples across genders to reduce gender bias."
        ),
        "fitness_level": (
            "Add more diverse training examples across fitness levels "
            "(beginner / intermediate / advanced) to close the performance gap."
        ),
        "goal_type": (
            "Ensure training conversations cover all fitness goal types "
            "proportionally; consider targeted data collection for lagging goals."
        ),
        "bmi_category": (
            "Review training data for BMI-category diversity; augment with "
            "examples spanning underweight, normal, overweight, and obese profiles."
        ),
    }

    for dim, slices in dim_issues.items():
        slice_list = ", ".join(sorted(slices))
        base = dim_advice.get(
            dim,
            f"Investigate training data balance for '{dim}' dimension.",
        )
        recommendations.append(
            f"[{dim}] Slices {slice_list} are flagged. {base}"
        )

    # Metric-level recommendations
    metric_advice: dict[str, str] = {
        "tool_call_accuracy": (
            "Low tool call accuracy in some slices suggests the model may not "
            "generalise tool selection across demographics. Consider targeted "
            "RLHF or DPO fine-tuning on slices with low accuracy."
        ),
        "json_parse_rate": (
            "Low JSON parse rate indicates formatting failures for certain "
            "demographic groups. Add output-format constraints or post-processing."
        ),
        "schema_compliance": (
            "Schema non-compliance in flagged slices may indicate inconsistent "
            "structured-output adherence. Strengthen system-prompt formatting "
            "instructions and add schema-guided decoding."
        ),
        "thinking_presence_rate": (
            "Uneven thinking-block presence suggests the model suppresses chain-"
            "of-thought reasoning for certain user profiles. Investigate if "
            "demographic cues in the prompt affect reasoning behaviour."
        ),
        "avg_response_length": (
            "Significant response-length variation across slices may indicate "
            "verbosity bias. Normalise expected output lengths in training data."
        ),
    }

    flagged_metrics = set(e["metric"] for e in flagged_slices)
    for metric in flagged_metrics:
        advice = metric_advice.get(
            metric,
            f"Investigate why '{metric}' shows disparity across demographic slices.",
        )
        recommendations.append(f"[{metric}] {advice}")

    return recommendations


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_bias_heatmaps(
    sliced_metrics: dict[str, dict[str, dict[str, float]]],
    overall_metrics: dict[str, float],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate heatmap visualisations of per-slice metric deviations.

    Produces:
    - ``plots/bias_heatmap.png``: summary heatmap with dimensions on the
      y-axis and metrics on the x-axis, cells coloured by deviation.
    - ``plots/<dimension>_heatmap.png``: one heatmap per dimension showing
      actual metric values per slice.

    Skips gracefully if matplotlib / seaborn are unavailable.

    Args:
        sliced_metrics: Nested dict: dimension -> slice_label -> metrics.
        overall_metrics: Overall metric dict from compute_overall_metrics().
        output_dir: Base output directory; plots/ subdirectory is created.
        logger: Logger instance.
    """
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]
        import numpy as np  # type: ignore[import]
        import seaborn as sns  # type: ignore[import]
    except ImportError as exc:
        logger.warning(
            "matplotlib/seaborn/numpy not available — skipping heatmaps: %s", exc
        )
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = _METRICS  # ordered list for consistent axes

    # ------------------------------------------------------------------
    # 1. Summary heatmap — max deviation per (dimension, metric)
    # ------------------------------------------------------------------
    dimensions = list(sliced_metrics.keys())
    summary_data: list[list[float]] = []

    for dim in dimensions:
        row_deviations: list[float] = []
        for metric in metric_cols:
            overall_val = overall_metrics.get(metric, 0.0)
            max_dev = 0.0
            for slice_m in sliced_metrics[dim].values():
                slice_val = slice_m.get(metric, 0.0)
                if overall_val == 0.0:
                    dev = 0.0 if slice_val == 0.0 else 1.0
                else:
                    dev = abs(slice_val - overall_val) / overall_val
                max_dev = max(max_dev, dev)
            row_deviations.append(round(max_dev, 4))
        summary_data.append(row_deviations)

    if dimensions and summary_data:
        import numpy as np  # already imported above, re-import avoids linter warning

        arr = np.array(summary_data)
        fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 1.5), max(4, len(dimensions) * 0.8)))
        sns.heatmap(
            arr,
            annot=True,
            fmt=".3f",
            xticklabels=metric_cols,
            yticklabels=dimensions,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=0.5,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Max Slice Deviation from Overall Mean (per Dimension × Metric)")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Dimension")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        summary_path = plots_dir / "bias_heatmap.png"
        fig.savefig(str(summary_path))
        plt.close(fig)
        logger.info("Saved summary heatmap: %s", summary_path)

    # ------------------------------------------------------------------
    # 2. Per-dimension heatmaps — actual metric values per slice
    # ------------------------------------------------------------------
    for dim, slices in sliced_metrics.items():
        slice_labels = sorted(slices.keys())
        if not slice_labels:
            continue

        dim_data: list[list[float]] = []
        for sl in slice_labels:
            dim_data.append([slices[sl].get(m, 0.0) for m in metric_cols])

        import numpy as np  # noqa: F811

        arr = np.array(dim_data)
        fig, ax = plt.subplots(
            figsize=(max(8, len(metric_cols) * 1.5), max(3, len(slice_labels) * 0.8))
        )
        sns.heatmap(
            arr,
            annot=True,
            fmt=".3f",
            xticklabels=metric_cols,
            yticklabels=slice_labels,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"Metric Values by Slice — {dim}")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Slice")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        dim_path = plots_dir / f"{dim}_heatmap.png"
        fig.savefig(str(dim_path))
        plt.close(fig)
        logger.info("Saved dimension heatmap: %s", dim_path)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_bias_report(
    output_dir: Path,
    report: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Write bias_report.json to the output directory.

    Args:
        output_dir: Directory to write the report into.
        report: Complete bias report dict.
        logger: Logger instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "bias_report.json"
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Bias report written to: %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the bias detection script."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect demographic bias in a fine-tuned QLoRA tool-calling LLM "
            "by slicing the validation set and comparing per-slice metrics."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Model-Pipeline/config/training_config.yaml",
        help="Path to training_config.yaml (default: Model-Pipeline/config/training_config.yaml)",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter directory (e.g. Model-Pipeline/outputs/final_adapter)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Model-Pipeline/outputs/bias_detection",
        help="Directory to write bias report and plots (default: Model-Pipeline/outputs/bias_detection)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help=(
            "Fractional deviation threshold for flagging a slice as biased "
            "(default: 0.1 = 10%%)"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of validation samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full bias detection pipeline."""
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("fitsense.bias_detection", level=log_level)

    # 1. Load config
    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Adapter dir:  %s", args.adapter_dir)
    logger.info("Output dir:   %s", output_dir)
    logger.info("Threshold:    %.2f", args.threshold)
    logger.info("Max samples:  %s", args.max_samples if args.max_samples else "all")
    logger.info("Git commit:   %s", get_git_commit() or "unavailable")

    # 2. Load validation dataset
    # Lazy import: load_data has no heavy deps so this is just for symmetry
    from load_data import load_and_validate  # type: ignore[import]

    train_path = config["train_path"]
    val_path = config["val_path"]
    logger.info("Loading validation data from: %s", val_path)
    datasets = load_and_validate(train_path, val_path, logger)
    val_dataset = datasets["validation"]

    n_samples = (
        min(args.max_samples, len(val_dataset))
        if args.max_samples is not None
        else len(val_dataset)
    )
    logger.info("Validation set size: %d (evaluating %d)", len(val_dataset), n_samples)

    # 3. Load model + adapter (lazy: triggers unsloth/torch only here)
    from evaluate import load_model_for_eval  # type: ignore[import]

    model, tokenizer = load_model_for_eval(args.adapter_dir, config, logger)

    # 4. Run evaluation with slice-attribute collection
    logger.info("Starting bias evaluation...")
    per_sample_results = run_bias_evaluation(
        model, tokenizer, val_dataset, args.max_samples, logger
    )

    # 5. Compute overall metrics
    logger.info("Computing overall metrics...")
    try:
        overall_metrics = compute_overall_metrics(per_sample_results)
    except ValueError as exc:
        logger.error("Overall metric computation failed: %s", exc)
        raise

    logger.info("Overall metrics:")
    for key, val in overall_metrics.items():
        logger.info("  %-32s %s", key + ":", val)

    # 6. Compute per-slice metrics for each dimension
    dimensions = ["age_group", "gender", "fitness_level", "goal_type", "bmi_category"]
    logger.info("Computing slice metrics across %d dimensions...", len(dimensions))

    sliced_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for dim in dimensions:
        sliced_metrics[dim] = compute_slice_metrics(per_sample_results, dim)
        logger.debug(
            "Dimension '%s': %d slices found",
            dim,
            len(sliced_metrics[dim]),
        )

    # 7. Detect bias
    logger.info("Detecting bias with threshold=%.2f...", args.threshold)
    flagged_slices = detect_bias(overall_metrics, sliced_metrics, args.threshold)

    bias_detected = len(flagged_slices) > 0
    logger.info(
        "Bias detection complete — %d flagged slices, bias_detected=%s",
        len(flagged_slices),
        bias_detected,
    )

    if flagged_slices:
        logger.warning("Flagged slices (top 10):")
        for entry in flagged_slices[:10]:
            logger.warning(
                "  [%s / %s] %s: slice=%.4f overall=%.4f deviation=%.4f",
                entry["dimension"],
                entry["slice"],
                entry["metric"],
                entry["slice_value"],
                entry["overall_value"],
                entry["deviation"],
            )

    # 8. Generate mitigation recommendations
    recommendations = generate_mitigation_recommendations(flagged_slices)
    logger.info("Mitigation recommendations (%d):", len(recommendations))
    for rec in recommendations:
        logger.info("  • %s", rec)

    # 9. Assemble the bias report
    bias_report: dict[str, Any] = {
        "model_name": config.get("model_name", "unknown"),
        "adapter_dir": str(args.adapter_dir),
        "n_samples": len(per_sample_results),
        "threshold": args.threshold,
        "overall_metrics": overall_metrics,
        "sliced_metrics": sliced_metrics,
        "flagged_slices": flagged_slices,
        "bias_detected": bias_detected,
        "mitigation_recommendations": recommendations,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 10. Write bias report
    write_bias_report(output_dir, bias_report, logger)

    # 11. Generate heatmap visualisations
    logger.info("Generating heatmap visualisations...")
    plot_bias_heatmaps(sliced_metrics, overall_metrics, output_dir, logger)

    logger.info("Bias detection pipeline complete. Outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
