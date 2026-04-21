"""
Sensitivity analysis for the FitSenseAI fine-tuned QLoRA adapter.

Performs two types of analysis:

1. **Hyperparameter sensitivity** — loads results from prior hparam search
   trials (all_trials.json), groups trials by each hyperparameter value, and
   computes a sensitivity score (range of mean eval losses) for each hparam.

2. **Input feature sensitivity** — applies perturbations to validation
   prompts (query length truncation, system prompt removal, profile info
   masking) and measures how output quality degrades relative to a baseline.

Writes sensitivity_report.json and matplotlib plots to the output directory.

Usage:
    python sensitivity.py \\
        --adapter-dir Model-Pipeline/outputs/final_adapter \\
        --config Model-Pipeline/config/training_config.yaml \\
        --output-dir Model-Pipeline/outputs/sensitivity

    # skip input sensitivity (no GPU):
    python sensitivity.py --adapter-dir ... --skip-input

    # skip hparam sensitivity (no trials file):
    python sensitivity.py --adapter-dir ... --skip-hparam
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Sibling-module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# Self-contained helpers re-declared here so that importing evaluate.py /
# train.py does NOT trigger their module-level heavy imports (wandb, trl,
# unsloth) at startup.  The actual evaluate/train helpers are imported lazily
# inside the functions that need them.

import yaml  # noqa: E402 — after sys.path manipulation


# ---------------------------------------------------------------------------
# Logging helpers (self-contained; mirrors train.setup_logger)
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


# ---------------------------------------------------------------------------
# Config helpers (self-contained; mirrors train.load_config)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Git helpers (self-contained; mirrors train.get_git_commit)
# ---------------------------------------------------------------------------

def get_git_commit() -> str | None:
    """Return the current HEAD commit SHA, or None if unavailable."""
    import subprocess  # lazy import

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
# Part 1: Hyperparameter sensitivity
# ---------------------------------------------------------------------------

def load_trials(trials_path: str) -> list[dict[str, Any]]:
    """Load and filter completed trials from all_trials.json.

    Only trials whose ``state`` is ``"COMPLETE"`` and that have a numeric
    ``value`` field are returned.

    Args:
        trials_path: Path to all_trials.json produced by hparam_search.py.

    Returns:
        List of completed trial dicts, each containing at minimum
        ``"value"`` (eval loss) and ``"params"`` (dict of hparam values).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If no completed trials are found.
    """
    path = Path(trials_path)
    if not path.exists():
        raise FileNotFoundError(f"Trials file not found: {trials_path}")

    with path.open("r") as fh:
        raw: list[dict[str, Any]] = json.load(fh)

    completed = [
        t for t in raw
        if t.get("state") == "COMPLETE" and isinstance(t.get("value"), (int, float))
    ]

    if not completed:
        raise ValueError(
            f"No completed trials found in {trials_path}. "
            f"Total records: {len(raw)}"
        )

    return completed


def analyze_hparam_sensitivity(
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-hparam sensitivity scores from completed trial results.

    For each hyperparameter the function groups trials by unique values of
    that hparam, computes the mean eval loss for each group, and derives a
    sensitivity score as (max_mean - min_mean).  A higher score means the
    hparam has more influence on the outcome.

    Args:
        trials: List of completed trial dicts from :func:`load_trials`.

    Returns:
        Dict mapping hparam_name -> {
            "values": [...],
            "mean_losses": [...],
            "sensitivity_score": float,
        }.

    Raises:
        ValueError: If trials list is empty.
    """
    if not trials:
        raise ValueError("analyze_hparam_sensitivity: trials list is empty")

    # Collect all hparam names from the first trial that has params
    all_hparam_names: set[str] = set()
    for t in trials:
        all_hparam_names.update(t.get("params", {}).keys())

    analysis: dict[str, Any] = {}

    for hparam in sorted(all_hparam_names):
        # Group losses by hparam value
        groups: dict[Any, list[float]] = defaultdict(list)
        for t in trials:
            val = t.get("params", {}).get(hparam)
            if val is None:
                continue
            groups[val].append(float(t["value"]))

        if not groups:
            continue

        # Sort by hparam value for deterministic output
        try:
            sorted_values = sorted(groups.keys())
        except TypeError:
            # Un-sortable types (mixed int/str) — fall back to insertion order
            sorted_values = list(groups.keys())

        mean_losses = [
            sum(groups[v]) / len(groups[v]) for v in sorted_values
        ]
        sensitivity_score = max(mean_losses) - min(mean_losses)

        analysis[hparam] = {
            "values": sorted_values,
            "mean_losses": [round(ml, 6) for ml in mean_losses],
            "sensitivity_score": round(sensitivity_score, 6),
        }

    return analysis


def rank_hparams(hparam_analysis: dict[str, Any]) -> list[str]:
    """Rank hyperparameters by sensitivity score, highest first.

    Args:
        hparam_analysis: Output of :func:`analyze_hparam_sensitivity`.

    Returns:
        List of hparam names sorted by descending sensitivity score.
    """
    return sorted(
        hparam_analysis.keys(),
        key=lambda k: hparam_analysis[k]["sensitivity_score"],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Part 2: Input feature sensitivity — text perturbation helpers
# ---------------------------------------------------------------------------

def truncate_message(text: str, fraction: float) -> str:
    """Truncate *text* to *fraction* of its original word count.

    Splitting is done on whitespace.  The truncation always yields at least
    one word so that an empty string is never returned for non-empty input.

    Args:
        text: Original message text.
        fraction: Fraction of words to keep (e.g. 0.5 for 50 %).

    Returns:
        Truncated string, joined with single spaces.

    Raises:
        ValueError: If fraction is not in (0, 1].
    """
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    words = text.split()
    if not words:
        return text

    keep = max(1, int(len(words) * fraction))
    return " ".join(words[:keep])


def mask_profile_info(text: str) -> str:
    """Remove demographic / profile mentions from a user message.

    Matches and removes references to:
    - Age patterns: "25 years old", "age 30", "I am 45", "I'm 28", "35yo",
      "35 yo", "35-year-old", "35 year old"
    - Gender identifiers: female, male, woman, man, girl, boy (whole-word)
    - Fitness levels: beginner, intermediate, advanced, sedentary (whole-word)
    - BMI references: "BMI 22.5", "bmi of 24"
    - Goal / profile keywords: "fitness level", "fitness goal"

    Args:
        text: Original user message content.

    Returns:
        Text with matched patterns replaced by a single space, then
        collapsed to remove double spaces.
    """
    patterns: list[str] = [
        # Age: "25 years old", "25-year-old", "25 year old", "25yo", "25 yo"
        r"\b\d{1,3}\s*[-\s]?years?\s*[-\s]?old\b",
        r"\b\d{1,3}\s*yo\b",
        # Age phrasing: "I am 30", "I'm 30", "age 30"
        r"\b(?:i\s+am|i'm|age)\s+\d{1,3}\b",
        # Gender identifiers (whole-word)
        r"\b(?:female|male|woman|man|girl|boy)\b",
        # Fitness level labels (whole-word)
        r"\b(?:beginner|intermediate|advanced|sedentary)\b",
        # BMI references
        r"\bbmi\s+(?:of\s+)?\d+(?:\.\d+)?\b",
        # Generic fitness profile phrases
        r"\bfitness\s+(?:level|goal)\b",
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)

    # Collapse multiple spaces
    result = re.sub(r" {2,}", " ", result).strip()
    return result


# ---------------------------------------------------------------------------
# Part 2: Generation metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics_from_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate generation results into summary metrics.

    Args:
        results: List of per-sample result dicts, each expected to have
            boolean fields ``json_parsed``, ``tool_call_correct``, and
            ``has_thinking``.

    Returns:
        Dict with keys ``tool_call_accuracy``, ``json_parse_rate``, and
        ``thinking_presence_rate``.
    """
    n = len(results)
    if n == 0:
        return {
            "tool_call_accuracy": 0.0,
            "json_parse_rate": 0.0,
            "thinking_presence_rate": 0.0,
        }

    tool_correct = sum(1 for r in results if r.get("tool_call_correct", False))
    n_with_ref = sum(1 for r in results if r.get("ref_tool_name") is not None)
    json_parsed = sum(1 for r in results if r.get("json_parsed", False))
    has_thinking = sum(1 for r in results if r.get("has_thinking", False))

    return {
        "tool_call_accuracy": round(tool_correct / n_with_ref, 6) if n_with_ref > 0 else 0.0,
        "json_parse_rate": round(json_parsed / n, 6),
        "thinking_presence_rate": round(has_thinking / n, 6),
    }


def compute_perturbation_metrics(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    perturbation_fn: Callable[[list[dict[str, str]]], list[dict[str, str]]],
    n_samples: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """Apply a message perturbation and compute generation quality metrics.

    For each of the first *n_samples* validation examples, the function:
    1. Applies *perturbation_fn* to the messages list.
    2. Generates a response using the model.
    3. Parses the response into reasoning + JSON answer.
    4. Checks JSON parseability and tool call correctness.

    Args:
        model: Inference-mode model loaded via Unsloth.
        tokenizer: Matching tokenizer.
        dataset: HuggingFace Dataset with a ``"messages"`` column.
        perturbation_fn: Callable that takes a messages list and returns a
            perturbed messages list.
        n_samples: Number of samples to evaluate.
        logger: Logger instance.

    Returns:
        Dict of aggregated metrics (tool_call_accuracy, json_parse_rate,
        thinking_presence_rate).
    """
    # Lazy imports to avoid triggering heavy dependencies at module load
    from evaluate import extract_answer_from_response, generate_single, parse_tool_name  # type: ignore[import]

    n = min(n_samples, len(dataset))
    results: list[dict[str, Any]] = []

    for idx in range(n):
        row = dataset[idx]
        messages: list[dict[str, str]] = list(row["messages"])

        # Extract reference tool name from ground-truth assistant turn
        ref_content = messages[2]["content"] if len(messages) >= 3 else ""
        _, ref_json_str = extract_answer_from_response(ref_content)
        ref_tool_name = parse_tool_name(ref_json_str) if ref_json_str else None

        # Apply perturbation
        try:
            perturbed_messages = perturbation_fn(messages)
        except Exception as exc:
            logger.warning("Sample %d: perturbation failed — %s", idx, exc)
            results.append({
                "ref_tool_name": ref_tool_name,
                "json_parsed": False,
                "tool_call_correct": False,
                "has_thinking": False,
            })
            continue

        # Generate response
        try:
            generated_text, _ = generate_single(model, tokenizer, perturbed_messages)
        except RuntimeError as exc:
            logger.warning("Sample %d: generation failed — %s", idx, exc)
            results.append({
                "ref_tool_name": ref_tool_name,
                "json_parsed": False,
                "tool_call_correct": False,
                "has_thinking": False,
            })
            continue

        # Parse output
        reasoning, json_str = extract_answer_from_response(generated_text)
        has_thinking = reasoning is not None
        json_parsed = False
        pred_tool_name: str | None = None

        if json_str:
            try:
                parsed = json.loads(json_str)
                json_parsed = True
                pred_tool_name = parsed.get("tool_name")
            except json.JSONDecodeError:
                pass

        results.append({
            "ref_tool_name": ref_tool_name,
            "json_parsed": json_parsed,
            "tool_call_correct": (
                ref_tool_name is not None and pred_tool_name == ref_tool_name
            ),
            "has_thinking": has_thinking,
        })

        if (idx + 1) % 10 == 0:
            logger.info("Perturbation eval: %d/%d samples", idx + 1, n)

    return _compute_metrics_from_results(results)


def run_input_sensitivity(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    n_samples: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run all input perturbation tests and return a structured results dict.

    Perturbations applied:
    - ``truncate_50pct`` — user message truncated to 50 % of word count
    - ``truncate_25pct`` — user message truncated to 25 % of word count
    - ``no_system_prompt`` — system message replaced with an empty string
    - ``mask_profile_info`` — age / gender / fitness-level mentions removed
      from the user message

    For each perturbation the degradation relative to the unperturbed
    baseline is computed per metric.

    Args:
        model: Inference-mode model loaded via Unsloth.
        tokenizer: Matching tokenizer.
        dataset: HuggingFace Dataset with a ``"messages"`` column.
        n_samples: Number of samples to use for each perturbation.
        logger: Logger instance.

    Returns:
        Dict with keys ``"baseline"`` and ``"perturbations"``, where each
        perturbation entry contains the raw metric values and a
        ``"degradation"`` sub-dict showing signed delta from baseline.
    """
    # ---- Define perturbation functions ----------------------------------- #

    def _identity(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return messages unchanged (baseline)."""
        return msgs

    def _truncate_50pct(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        result = list(msgs)
        if len(result) >= 2:
            result[1] = {**result[1], "content": truncate_message(result[1]["content"], 0.5)}
        return result

    def _truncate_25pct(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        result = list(msgs)
        if len(result) >= 2:
            result[1] = {**result[1], "content": truncate_message(result[1]["content"], 0.25)}
        return result

    def _no_system_prompt(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        result = list(msgs)
        if result and result[0].get("role") == "system":
            result[0] = {**result[0], "content": ""}
        return result

    def _mask_profile(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        result = list(msgs)
        if len(result) >= 2:
            result[1] = {**result[1], "content": mask_profile_info(result[1]["content"])}
        return result

    perturbations: dict[str, Callable[[list[dict[str, str]]], list[dict[str, str]]]] = {
        "truncate_50pct": _truncate_50pct,
        "truncate_25pct": _truncate_25pct,
        "no_system_prompt": _no_system_prompt,
        "mask_profile_info": _mask_profile,
    }

    # ---- Baseline -------------------------------------------------------- #
    logger.info("Computing baseline metrics on %d samples...", n_samples)
    baseline = compute_perturbation_metrics(
        model, tokenizer, dataset, _identity, n_samples, logger
    )
    logger.info("Baseline: %s", baseline)

    # ---- Perturbations --------------------------------------------------- #
    perturbation_results: dict[str, Any] = {}

    for name, fn in perturbations.items():
        logger.info("Running perturbation: %s", name)
        metrics = compute_perturbation_metrics(
            model, tokenizer, dataset, fn, n_samples, logger
        )
        degradation = {
            metric: round(metrics[metric] - baseline[metric], 6)
            for metric in baseline
        }
        perturbation_results[name] = {**metrics, "degradation": degradation}
        logger.info("  metrics: %s", metrics)
        logger.info("  degradation: %s", degradation)

    return {
        "baseline": baseline,
        "perturbations": perturbation_results,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hparam_sensitivity(
    hparam_analysis: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate hyperparameter sensitivity plots.

    Produces:
    - ``plots/hparam_sensitivity_ranking.png`` — horizontal bar chart of
      hparams ranked by sensitivity score.
    - ``plots/hparam_<name>.png`` — scatter/line plot of mean loss vs.
      hparam value for each hyperparameter.

    Skips gracefully if matplotlib is unavailable.

    Args:
        hparam_analysis: Output of :func:`analyze_hparam_sensitivity`.
        output_dir: Base output directory; a ``plots/`` subdirectory is
            created inside it.
        logger: Logger instance.
    """
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:
        logger.warning("matplotlib not available — skipping hparam plots")
        return

    if not hparam_analysis:
        logger.warning("hparam_analysis is empty — skipping hparam plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ranking = rank_hparams(hparam_analysis)
    scores = [hparam_analysis[h]["sensitivity_score"] for h in ranking]

    # 1. Horizontal bar chart — ranking
    fig, ax = plt.subplots(figsize=(8, max(4, len(ranking) * 0.6)))
    ax.barh(ranking[::-1], scores[::-1], color="steelblue")
    ax.set_xlabel("Sensitivity Score (range of mean eval losses)")
    ax.set_title("Hyperparameter Sensitivity Ranking")
    for i, (name, score) in enumerate(zip(reversed(ranking), reversed(scores))):
        ax.text(score + 0.001, i, f"{score:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    ranking_path = plots_dir / "hparam_sensitivity_ranking.png"
    fig.savefig(str(ranking_path))
    plt.close(fig)
    logger.info("Saved: %s", ranking_path)

    # 2. Per-hparam scatter/line plot
    for hparam, info in hparam_analysis.items():
        values = info["values"]
        mean_losses = info["mean_losses"]

        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            # Plot as a line if values are numeric and more than one point
            if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
                ax.plot(values, mean_losses, marker="o", color="steelblue", linewidth=1.5)
            else:
                x_pos = list(range(len(values)))
                ax.scatter(x_pos, mean_losses, color="steelblue", zorder=3)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(v) for v in values], rotation=30, ha="right")
        except (TypeError, ValueError) as exc:
            logger.warning("Could not plot hparam '%s': %s", hparam, exc)
            plt.close(fig)
            continue

        ax.set_xlabel(hparam)
        ax.set_ylabel("Mean Eval Loss")
        ax.set_title(f"Eval Loss vs {hparam}  (sensitivity={info['sensitivity_score']:.4f})")
        fig.tight_layout()
        hp_path = plots_dir / f"hparam_{hparam}.png"
        fig.savefig(str(hp_path))
        plt.close(fig)
        logger.info("Saved: %s", hp_path)


def plot_input_sensitivity(
    input_results: dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate the input perturbation impact grouped bar chart.

    Produces ``plots/input_perturbation_impact.png`` — a grouped bar chart
    with one group per metric, showing baseline and each perturbation side
    by side.

    Skips gracefully if matplotlib is unavailable or results are empty.

    Args:
        input_results: Output of :func:`run_input_sensitivity`.
        output_dir: Base output directory; a ``plots/`` subdirectory is
            created inside it.
        logger: Logger instance.
    """
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError:
        logger.warning("matplotlib/numpy not available — skipping input sensitivity plot")
        return

    if not input_results:
        logger.warning("input_results is empty — skipping input sensitivity plot")
        return

    baseline = input_results.get("baseline", {})
    perturbations = input_results.get("perturbations", {})

    if not baseline or not perturbations:
        logger.warning("No baseline or perturbations data — skipping input sensitivity plot")
        return

    metrics = list(baseline.keys())
    perturbation_names = list(perturbations.keys())

    # Build value matrix: rows = metrics, cols = [baseline] + perturbations
    all_names = ["baseline"] + perturbation_names
    n_groups = len(metrics)
    n_bars = len(all_names)

    values_matrix: list[list[float]] = []
    for metric in metrics:
        row = [baseline.get(metric, 0.0)]
        for pname in perturbation_names:
            row.append(perturbations[pname].get(metric, 0.0))
        values_matrix.append(row)

    x = np.arange(n_groups)
    width = 0.8 / n_bars
    offsets = np.linspace(-(0.8 / 2) + width / 2, (0.8 / 2) - width / 2, n_bars)

    colors = ["steelblue", "coral", "seagreen", "mediumpurple", "goldenrod"]

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.8), 5))
    for i, (name, offset) in enumerate(zip(all_names, offsets)):
        bar_values = [values_matrix[j][i] for j in range(n_groups)]
        color = colors[i % len(colors)]
        bars = ax.bar(x + offset, bar_values, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, bar_values):
            if val != 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Input Perturbation Impact on Generation Metrics")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    impact_path = plots_dir / "input_perturbation_impact.png"
    fig.savefig(str(impact_path))
    plt.close(fig)
    logger.info("Saved: %s", impact_path)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    hparam_results: dict[str, Any] | None,
    input_results: dict[str, Any] | None,
    logger: logging.Logger,
) -> Path:
    """Assemble and write sensitivity_report.json.

    Args:
        output_dir: Directory to write the report into.
        hparam_results: Output of :func:`analyze_hparam_sensitivity` with a
            ``"ranking"`` key added, or None if hparam sensitivity was skipped.
        input_results: Output of :func:`run_input_sensitivity`, or None if
            input sensitivity was skipped.
        logger: Logger instance.

    Returns:
        Path to the written report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "hparam_sensitivity": hparam_results,
        "input_sensitivity": input_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
    }

    report_path = output_dir / "sensitivity_report.json"
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Sensitivity report written to: %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sensitivity analysis script."""
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for the FitSenseAI QLoRA adapter"
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
        help="Path to the saved LoRA adapter directory (required for input sensitivity)",
    )
    parser.add_argument(
        "--trials-file",
        type=str,
        default="Model-Pipeline/outputs/hparam_search/all_trials.json",
        help=(
            "Path to all_trials.json from hparam search "
            "(default: Model-Pipeline/outputs/hparam_search/all_trials.json)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Model-Pipeline/outputs/sensitivity",
        help="Directory to write sensitivity results (default: Model-Pipeline/outputs/sensitivity)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of validation samples for input perturbation tests (default: 50)",
    )
    parser.add_argument(
        "--skip-hparam",
        action="store_true",
        default=False,
        help="Skip hyperparameter sensitivity analysis (e.g. when no trials file exists)",
    )
    parser.add_argument(
        "--skip-input",
        action="store_true",
        default=False,
        help="Skip input feature sensitivity analysis (e.g. when no GPU is available)",
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
    """Orchestrate hyperparameter and input feature sensitivity analysis."""
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("fitsense.sensitivity", level=log_level)

    # 1. Load config
    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Adapter dir:  %s", args.adapter_dir)
    logger.info("Trials file:  %s", args.trials_file)
    logger.info("Output dir:   %s", output_dir)
    logger.info("N samples:    %d", args.n_samples)
    logger.info("Skip hparam:  %s", args.skip_hparam)
    logger.info("Skip input:   %s", args.skip_input)
    logger.info("Git commit:   %s", get_git_commit() or "unavailable")

    hparam_results: dict[str, Any] | None = None
    input_results: dict[str, Any] | None = None

    # ---------------------------------------------------------------------- #
    # Part 1: Hyperparameter sensitivity
    # ---------------------------------------------------------------------- #
    if args.skip_hparam:
        logger.info("Skipping hyperparameter sensitivity (--skip-hparam set)")
    else:
        logger.info("=" * 60)
        logger.info("Part 1: Hyperparameter Sensitivity")
        logger.info("=" * 60)
        try:
            trials = load_trials(args.trials_file)
            logger.info("Loaded %d completed trials from: %s", len(trials), args.trials_file)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Could not load trials: %s", exc)
            logger.warning("Skipping hyperparameter sensitivity due to load error.")
            trials = []

        if trials:
            try:
                analysis = analyze_hparam_sensitivity(trials)
                ranking = rank_hparams(analysis)
                hparam_results = {**analysis, "ranking": ranking}

                logger.info("Hyperparameter sensitivity ranking:")
                for hparam in ranking:
                    score = analysis[hparam]["sensitivity_score"]
                    logger.info("  %-20s sensitivity_score=%.6f", hparam, score)

                plot_hparam_sensitivity(analysis, output_dir, logger)
            except ValueError as exc:
                logger.error("Hyperparameter sensitivity analysis failed: %s", exc)

    # ---------------------------------------------------------------------- #
    # Part 2: Input feature sensitivity
    # ---------------------------------------------------------------------- #
    if args.skip_input:
        logger.info("Skipping input feature sensitivity (--skip-input set)")
    else:
        logger.info("=" * 60)
        logger.info("Part 2: Input Feature Sensitivity")
        logger.info("=" * 60)

        # Lazy imports — only pulled in when input sensitivity is actually run
        try:
            from evaluate import load_model_for_eval  # type: ignore[import]
            from load_data import load_and_validate  # type: ignore[import]
        except ImportError as exc:
            logger.error(
                "Could not import evaluation helpers: %s. "
                "Skipping input feature sensitivity.",
                exc,
            )
        else:
            # Load validation dataset
            try:
                datasets = load_and_validate(
                    config["train_path"], config["val_path"], logger
                )
                val_dataset = datasets["validation"]
                logger.info(
                    "Validation set loaded — %d samples (using %d)",
                    len(val_dataset),
                    min(args.n_samples, len(val_dataset)),
                )
            except (FileNotFoundError, KeyError) as exc:
                logger.error("Failed to load validation data: %s", exc)
                val_dataset = None

            if val_dataset is not None:
                # Load model + adapter
                try:
                    model, tokenizer = load_model_for_eval(
                        args.adapter_dir, config, logger
                    )
                except (ImportError, RuntimeError) as exc:
                    logger.error("Failed to load model for eval: %s", exc)
                    model, tokenizer = None, None

                if model is not None and tokenizer is not None:
                    try:
                        input_results = run_input_sensitivity(
                            model, tokenizer, val_dataset, args.n_samples, logger
                        )
                        plot_input_sensitivity(input_results, output_dir, logger)
                    except Exception as exc:
                        logger.error(
                            "Input feature sensitivity analysis failed: %s", exc,
                            exc_info=True,
                        )

    # ---------------------------------------------------------------------- #
    # Write report
    # ---------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("Writing sensitivity report")
    logger.info("=" * 60)
    report_path = write_report(output_dir, hparam_results, input_results, logger)

    logger.info("Sensitivity analysis complete. Report: %s", report_path)
    logger.info("Outputs directory: %s", output_dir)


if __name__ == "__main__":
    main()
