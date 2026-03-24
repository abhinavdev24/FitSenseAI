"""
Evaluation script for the FitSenseAI fine-tuned QLoRA adapter.

Loads a base Qwen3 model with a trained LoRA adapter and evaluates it on the
validation set. Computes metrics specific to tool-calling with chain-of-thought
distillation: validation loss, tool call accuracy, JSON parse rate, schema
compliance, thinking presence and quality, and response latency.

Writes evaluation_results.json, per_sample_results.jsonl, and plots to the
specified output directory.

Usage:
    python evaluate.py \\
        --adapter-dir Model-Pipeline/outputs/final_adapter \\
        --config Model-Pipeline/config/training_config.yaml \\
        --output-dir Model-Pipeline/outputs/evaluation \\
        --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sibling-module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_and_validate
from train import get_git_commit, load_config, setup_logger


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_eval(
    adapter_dir: str,
    config: dict[str, Any],
    logger: logging.Logger,
):
    """Load the base model + LoRA adapter and enable Unsloth inference mode.

    Args:
        adapter_dir: Path to the saved LoRA adapter directory.
        config: Training/eval config dictionary (needs max_seq_length).
        logger: Logger instance.

    Returns:
        Tuple of (model, tokenizer) ready for inference.

    Raises:
        ImportError: If unsloth is not installed.
        RuntimeError: If model loading fails.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "unsloth is required for model loading. "
            "Install it with: pip install unsloth"
        ) from exc

    logger.info("Loading model + adapter from: %s", adapter_dir)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_dir,
            max_seq_length=config["max_seq_length"],
            load_in_4bit=True,
            dtype=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from '{adapter_dir}': {exc}"
        ) from exc

    FastLanguageModel.for_inference(model)
    logger.info("Model loaded and inference mode enabled")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------

def extract_answer_from_response(text: str) -> tuple[str | None, str | None]:
    """Split a model response into reasoning and JSON answer portions.

    If the response contains a ``</think>`` tag the content before it is the
    reasoning trace and the content after is the JSON answer string.  When no
    ``</think>`` tag is present the entire text is treated as the JSON answer.

    Args:
        text: Raw decoded text from the model.

    Returns:
        Tuple of (reasoning, json_str).  Either element may be None when the
        corresponding portion is absent or empty.
    """
    if "</think>" in text:
        parts = text.split("</think>", maxsplit=1)
        reasoning = parts[0].replace("<think>", "").strip()
        json_str = parts[1].strip()
        return (reasoning or None, json_str or None)
    return (None, text.strip() or None)


def parse_tool_name(json_str: str) -> str | None:
    """Attempt to parse the JSON answer and extract the tool_name field.

    Args:
        json_str: String that should contain a JSON object.

    Returns:
        The value of the ``tool_name`` key, or None if parsing fails or the
        key is absent.
    """
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
        return data.get("tool_name")
    except (json.JSONDecodeError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_single(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 4096,
) -> tuple[str, float]:
    """Generate a single response for a list of messages.

    Takes only the system + user messages (first two turns), formats them with
    the chat template, calls model.generate(), and decodes only the newly
    generated tokens.

    Args:
        model: Inference-mode model loaded via Unsloth.
        tokenizer: Matching tokenizer.
        messages: Full messages list; only the first two entries are used.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Tuple of (generated_text, latency_ms) where latency_ms is the
        wall-clock time for model.generate() in milliseconds.

    Raises:
        RuntimeError: If tokenisation or generation fails.
    """
    import torch  # lazy import

    # Use only system + user messages as the prompt
    prompt_messages = messages[:2]

    try:
        input_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception as exc:
        raise RuntimeError(f"Chat template application failed: {exc}") from exc

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    try:
        t0 = time.monotonic()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
            )
        latency_ms = (time.monotonic() - t0) * 1000.0
    except Exception as exc:
        raise RuntimeError(f"model.generate() failed: {exc}") from exc

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text, latency_ms


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

def compute_val_loss(
    model,
    tokenizer,
    dataset,
    logger: logging.Logger,
) -> float:
    """Compute average cross-entropy validation loss over the dataset.

    Uses a forward pass (not generation) for efficiency.  Sequences that are
    too long to tokenise are skipped with a warning.

    Args:
        model: Inference-mode model.
        tokenizer: Matching tokenizer.
        dataset: HuggingFace Dataset with 'messages' column.
        logger: Logger instance.

    Returns:
        Average cross-entropy loss across all processed samples.

    Raises:
        RuntimeError: If no samples could be processed.
    """
    import torch  # lazy import

    total_loss = 0.0
    n_processed = 0

    model.eval()
    device = next(model.parameters()).device

    for idx, row in enumerate(dataset):
        messages: list[dict[str, str]] = row["messages"]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(device)
        except Exception as exc:
            logger.warning("Row %d: tokenisation failed — %s", idx, exc)
            continue

        labels = input_ids.clone()

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
            loss_val = outputs.loss.item()
        except Exception as exc:
            logger.warning("Row %d: forward pass failed — %s", idx, exc)
            continue

        total_loss += loss_val
        n_processed += 1

    if n_processed == 0:
        raise RuntimeError("compute_val_loss: no samples were processed successfully")

    avg_loss = total_loss / n_processed
    logger.info(
        "Validation loss computed over %d samples: %.6f", n_processed, avg_loss
    )
    return avg_loss


# ---------------------------------------------------------------------------
# Generation evaluation loop
# ---------------------------------------------------------------------------

def evaluate_generation(
    model,
    tokenizer,
    dataset,
    max_samples: int | None,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Generate responses for the validation set and collect per-sample results.

    For each sample the reference tool_name is extracted from
    ``messages[2]["content"]`` (the ground-truth assistant turn).  Generation
    uses only ``messages[:2]`` as the prompt.

    Args:
        model: Inference-mode model.
        tokenizer: Matching tokenizer.
        dataset: HuggingFace Dataset with 'messages' column.
        max_samples: If provided, evaluate only this many samples.
        logger: Logger instance.

    Returns:
        List of per-sample result dicts, one per evaluated sample.
    """
    results: list[dict[str, Any]] = []
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    logger.info("Evaluating generation on %d samples", n_samples)

    for idx in range(n_samples):
        row = dataset[idx]
        messages: list[dict[str, str]] = row["messages"]

        # --- Extract reference tool name from ground truth ---
        ref_content = messages[2]["content"] if len(messages) >= 3 else ""
        _, ref_json_str = extract_answer_from_response(ref_content)
        ref_tool_name = parse_tool_name(ref_json_str) if ref_json_str else None

        # --- Generate ---
        try:
            generated_text, latency_ms = generate_single(model, tokenizer, messages)
        except RuntimeError as exc:
            logger.warning("Sample %d: generation failed — %s", idx, exc)
            results.append(
                {
                    "sample_idx": idx,
                    "ref_tool_name": ref_tool_name,
                    "pred_tool_name": None,
                    "json_parsed": False,
                    "schema_compliant": False,
                    "has_thinking": False,
                    "thinking_length": 0,
                    "latency_ms": 0.0,
                    "generated_text": "",
                }
            )
            continue

        # --- Parse generated output ---
        reasoning, json_str = extract_answer_from_response(generated_text)
        has_thinking = reasoning is not None
        thinking_length = len(reasoning) if reasoning else 0

        json_parsed = False
        schema_compliant = False
        pred_tool_name: str | None = None

        if json_str:
            try:
                parsed = json.loads(json_str)
                json_parsed = True
                pred_tool_name = parsed.get("tool_name")
                schema_compliant = (
                    "tool_name" in parsed and "tool_input" in parsed
                )
            except json.JSONDecodeError:
                pass

        results.append(
            {
                "sample_idx": idx,
                "ref_tool_name": ref_tool_name,
                "pred_tool_name": pred_tool_name,
                "json_parsed": json_parsed,
                "schema_compliant": schema_compliant,
                "has_thinking": has_thinking,
                "thinking_length": thinking_length,
                "latency_ms": round(latency_ms, 2),
                "generated_text": generated_text[:500],
            }
        )

        if (idx + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d samples evaluated", idx + 1, n_samples
            )

    logger.info("Generation evaluation complete — %d samples", len(results))
    return results


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(per_sample_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from per-sample result dicts.

    Args:
        per_sample_results: Output of evaluate_generation().

    Returns:
        Dictionary of aggregate metric values.

    Raises:
        ValueError: If per_sample_results is empty.
    """
    n = len(per_sample_results)
    if n == 0:
        raise ValueError("aggregate_metrics: per_sample_results is empty")

    correct_tool = sum(
        1
        for r in per_sample_results
        if r["ref_tool_name"] is not None
        and r["pred_tool_name"] == r["ref_tool_name"]
    )
    # Denominator for tool accuracy is samples where reference has a tool_name
    n_with_ref = sum(
        1 for r in per_sample_results if r["ref_tool_name"] is not None
    )

    json_parsed = sum(1 for r in per_sample_results if r["json_parsed"])
    schema_compliant = sum(1 for r in per_sample_results if r["schema_compliant"])
    has_thinking = sum(1 for r in per_sample_results if r["has_thinking"])

    thinking_lengths = [
        r["thinking_length"]
        for r in per_sample_results
        if r["has_thinking"] and r["thinking_length"] > 0
    ]
    avg_thinking_length = (
        sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0.0
    )

    latencies = [r["latency_ms"] for r in per_sample_results if r["latency_ms"] > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "tool_call_accuracy": round(correct_tool / n_with_ref, 6) if n_with_ref > 0 else 0.0,
        "json_parse_rate": round(json_parsed / n, 6),
        "schema_compliance": round(schema_compliant / max(json_parsed, 1), 6),
        "thinking_presence_rate": round(has_thinking / n, 6),
        "avg_thinking_length": round(avg_thinking_length, 2),
        "avg_response_latency_ms": round(avg_latency, 2),
    }


def aggregate_per_tool_accuracy(
    per_sample_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute tool-call accuracy broken down by reference tool name.

    Args:
        per_sample_results: Output of evaluate_generation().

    Returns:
        Dict mapping tool_name -> accuracy (float in [0, 1]).
    """
    tool_correct: dict[str, int] = defaultdict(int)
    tool_total: dict[str, int] = defaultdict(int)

    for r in per_sample_results:
        ref = r["ref_tool_name"]
        if ref is None:
            continue
        tool_total[ref] += 1
        if r["pred_tool_name"] == ref:
            tool_correct[ref] += 1

    return {
        tool: round(tool_correct[tool] / total, 6)
        for tool, total in tool_total.items()
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_results(
    output_dir: Path,
    eval_record: dict[str, Any],
    per_sample_results: list[dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Write evaluation_results.json and per_sample_results.jsonl to disk.

    Args:
        output_dir: Directory to write output files into.
        eval_record: Aggregate evaluation record (evaluation_results.json).
        per_sample_results: Per-sample result dicts (per_sample_results.jsonl).
        logger: Logger instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_results.json"
    with results_path.open("w") as fh:
        json.dump(eval_record, fh, indent=2)
    logger.info("Evaluation results written to: %s", results_path)

    per_sample_path = output_dir / "per_sample_results.jsonl"
    with per_sample_path.open("w") as fh:
        for record in per_sample_results:
            fh.write(json.dumps(record) + "\n")
    logger.info(
        "Per-sample results written to: %s (%d lines)",
        per_sample_path,
        len(per_sample_results),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(
    metrics: dict[str, Any],
    per_tool_accuracy: dict[str, float],
    per_sample_results: list[dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate evaluation visualisation plots.

    Produces three PNG files under ``output_dir/plots/``:
    - metrics_summary.png — bar chart of aggregate metrics
    - tool_accuracy_by_type.png — per-tool accuracy bar chart
    - latency_distribution.png — histogram of per-sample generation latency

    Skips gracefully if matplotlib is unavailable.

    Args:
        metrics: Aggregate metrics dict from aggregate_metrics().
        per_tool_accuracy: Per-tool accuracy dict from aggregate_per_tool_accuracy().
        per_sample_results: Per-sample result dicts from evaluate_generation().
        output_dir: Base output directory; plots/ subdirectory will be created.
        logger: Logger instance.
    """
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:
        logger.warning("matplotlib is not available — skipping plot generation")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics summary bar chart
    scalar_metrics = {
        k: v
        for k, v in metrics.items()
        if k not in ("avg_thinking_length", "avg_response_latency_ms")
        and isinstance(v, float)
    }
    if scalar_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(scalar_metrics.keys())
        values = [scalar_metrics[n] for n in names]
        bars = ax.bar(names, values, color="steelblue")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Metrics Summary")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        summary_path = plots_dir / "metrics_summary.png"
        fig.savefig(str(summary_path))
        plt.close(fig)
        logger.info("Saved: %s", summary_path)

    # 2. Per-tool accuracy bar chart
    if per_tool_accuracy:
        tools = sorted(per_tool_accuracy.keys())
        accs = [per_tool_accuracy[t] for t in tools]
        fig, ax = plt.subplots(figsize=(max(8, len(tools) * 0.9), 5))
        bars = ax.bar(tools, accs, color="seagreen")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Tool Call Accuracy by Tool Type")
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        tool_path = plots_dir / "tool_accuracy_by_type.png"
        fig.savefig(str(tool_path))
        plt.close(fig)
        logger.info("Saved: %s", tool_path)

    # 3. Latency distribution histogram
    latencies = [r["latency_ms"] for r in per_sample_results if r["latency_ms"] > 0]
    if latencies:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(latencies, bins=30, color="coral", edgecolor="black")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Per-Sample Response Latency Distribution")
        fig.tight_layout()
        latency_path = plots_dir / "latency_distribution.png"
        fig.savefig(str(latency_path))
        plt.close(fig)
        logger.info("Saved: %s", latency_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned QLoRA adapter on the FitSenseAI validation set"
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
        default="Model-Pipeline/outputs/evaluation",
        help="Directory to write evaluation results (default: Model-Pipeline/outputs/evaluation)",
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
    """Main evaluation entrypoint."""
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("fitsense.evaluate", level=log_level)

    # 1. Load config
    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Adapter dir: %s", args.adapter_dir)
    logger.info("Output dir:  %s", output_dir)
    logger.info("Max samples: %s", args.max_samples if args.max_samples else "all")
    logger.info("Git commit:  %s", get_git_commit() or "unavailable")

    # 2. Load validation dataset
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

    # 3. Load model + adapter
    model, tokenizer = load_model_for_eval(args.adapter_dir, config, logger)

    # 4. Compute validation loss (forward pass, no generation)
    logger.info("Computing validation loss via forward pass...")
    # Subset the dataset when max_samples is set to be consistent
    if args.max_samples is not None:

        val_subset = val_dataset.select(range(n_samples))
    else:
        val_subset = val_dataset

    try:
        val_loss = compute_val_loss(model, tokenizer, val_subset, logger)
    except RuntimeError as exc:
        logger.error("Validation loss computation failed: %s", exc)
        val_loss = float("nan")

    # 5. Generation evaluation loop
    logger.info("Starting generation evaluation...")
    per_sample_results = evaluate_generation(
        model, tokenizer, val_subset, args.max_samples, logger
    )

    # 6. Aggregate metrics
    logger.info("Aggregating metrics...")
    try:
        gen_metrics = aggregate_metrics(per_sample_results)
    except ValueError as exc:
        logger.error("Metric aggregation failed: %s", exc)
        gen_metrics = {}

    per_tool_accuracy = aggregate_per_tool_accuracy(per_sample_results)

    all_metrics: dict[str, Any] = {"val_loss": round(val_loss, 6)}
    all_metrics.update(gen_metrics)

    # 7. Log metrics summary
    logger.info("=" * 60)
    logger.info("Evaluation Metrics")
    logger.info("=" * 60)
    for key, value in all_metrics.items():
        logger.info("  %-32s %s", key + ":", value)
    logger.info("-" * 60)
    logger.info("Per-tool accuracy:")
    for tool, acc in sorted(per_tool_accuracy.items()):
        logger.info("  %-40s %.4f", tool + ":", acc)
    logger.info("=" * 60)

    # 8. Build the final evaluation record
    eval_record: dict[str, Any] = {
        "model_name": config.get("model_name", "unknown"),
        "adapter_dir": str(args.adapter_dir),
        "n_samples": len(per_sample_results),
        "metrics": all_metrics,
        "per_tool_accuracy": per_tool_accuracy,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 9. Write outputs
    write_results(output_dir, eval_record, per_sample_results, logger)

    # 10. Generate plots
    plot_results(all_metrics, per_tool_accuracy, per_sample_results, output_dir, logger)

    logger.info("Evaluation complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
