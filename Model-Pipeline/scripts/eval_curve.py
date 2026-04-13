"""
eval_curve.py

Post-hoc eval curve builder for FitSenseAI checkpoints.

Iterates over saved checkpoints, computes val loss sample-by-sample in bf16
(no full-logit materialisation), and logs to the same W&B run as training so
eval/loss appears alongside train/loss on a single chart.

Usage:
    python Model-Pipeline/scripts/eval_curve.py \
        --config Model-Pipeline/config/training_config.yaml \
        --wandb-run-id <run_id> \
        [--checkpoint-dir Model-Pipeline/outputs]

The W&B run ID is printed at the end of training, e.g.:
    wandb: 🚀 View run at .../runs/aun7qd4e   ← that's your run ID

Note: only checkpoints still on disk are evaluated. With save_total_limit=3
and save_steps=45 over 270 total steps, you will have at most 3 checkpoints
(steps 180, 225, 270). To get more points, increase save_total_limit before
the next training run.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_model_for_eval
from load_data import load_and_validate
from train import load_config, setup_logger


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(
    checkpoint_dir: Path,
    logger: logging.Logger,
) -> list[tuple[int, Path]]:
    """Return (step, path) pairs for all checkpoint-N subdirs, sorted by step.

    Args:
        checkpoint_dir: Directory that contains checkpoint-N subdirectories.
        logger: Logger instance.

    Returns:
        List of (step, path) tuples sorted ascending by step number.
    """
    checkpoints = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.removeprefix("checkpoint-"))
                checkpoints.append((step, d))
            except ValueError:
                logger.warning("Skipping unrecognised dir: %s", d.name)

    checkpoints.sort(key=lambda x: x[0])
    logger.info(
        "Found %d checkpoint(s) at steps: %s",
        len(checkpoints),
        [s for s, _ in checkpoints],
    )
    return checkpoints


# ---------------------------------------------------------------------------
# Memory-safe val loss
# ---------------------------------------------------------------------------

def compute_val_loss_safe(
    model,
    tokenizer,
    dataset,
    logger: logging.Logger,
) -> float:
    """Compute mean val loss sample-by-sample with CUDA cache clearing.

    Calls model(input_ids=..., labels=...) so cross-entropy is computed
    internally via a fused kernel — the full [batch, seq_len, vocab] logit
    tensor is never materialised in Python. Cache is cleared after every
    sample to avoid fragmentation across long sequences.

    Args:
        model: Inference-mode model loaded via Unsloth.
        tokenizer: Matching tokenizer.
        dataset: HuggingFace Dataset with a 'messages' column.
        logger: Logger instance.

    Returns:
        Mean cross-entropy loss over all successfully processed samples.

    Raises:
        RuntimeError: If no samples could be processed.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    n_ok = 0

    with torch.no_grad():
        for idx, row in enumerate(dataset):
            try:
                input_ids = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=True,
                    return_tensors="pt",
                ).to(device)
                outputs = model(input_ids=input_ids, labels=input_ids.clone())
                total_loss += outputs.loss.item()
                n_ok += 1
                del input_ids, outputs
            except Exception as exc:
                logger.warning("Row %d skipped: %s", idx, exc)
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            if (idx + 1) % 20 == 0:
                logger.info(
                    "  %d/%d samples — running avg loss: %.4f",
                    idx + 1,
                    len(dataset),
                    total_loss / max(n_ok, 1),
                )

    if n_ok == 0:
        raise RuntimeError("compute_val_loss_safe: no samples processed successfully")

    avg = total_loss / n_ok
    logger.info("Val loss: %.6f  (%d/%d samples)", avg, n_ok, len(dataset))
    return avg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc eval curve — computes val loss at each checkpoint "
            "and logs to an existing W&B run"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Model-Pipeline/config/training_config.yaml",
        help="Path to training_config.yaml",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help=(
            "W&B run ID to resume (e.g. aun7qd4e). "
            "If omitted, read from wandb_run_id.txt in output_dir."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory containing checkpoint-N subdirs. "
            "Defaults to output_dir from config."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    args = parse_args()
    logger = setup_logger(
        "fitsense.eval_curve", level=getattr(logging, args.log_level)
    )

    config = load_config(args.config)
    checkpoint_dir = Path(args.checkpoint_dir or config["output_dir"])

    # Resolve W&B run ID: CLI flag > wandb_run_id.txt > error
    run_id = args.wandb_run_id
    if not run_id:
        run_id_path = checkpoint_dir / "wandb_run_id.txt"
        if run_id_path.exists():
            run_id = run_id_path.read_text().strip()
            logger.info("Loaded W&B run ID from %s: %s", run_id_path, run_id)
        else:
            logger.error(
                "No --wandb-run-id given and %s not found.", run_id_path
            )
            sys.exit(1)

    # 1. Find checkpoints
    checkpoints = find_checkpoints(checkpoint_dir, logger)
    if not checkpoints:
        logger.error("No checkpoints found in: %s", checkpoint_dir)
        sys.exit(1)

    # 2. Load val dataset once — reused across all checkpoints
    logger.info("Loading val dataset from: %s", config["val_path"])
    datasets = load_and_validate(config["train_path"], config["val_path"], logger)
    val_dataset = datasets["validation"]
    logger.info("Val samples: %d", len(val_dataset))

    # 3. Resume the existing W&B training run
    logger.info("Resuming W&B run: %s", run_id)
    wandb.init(
        id=run_id,
        project=config.get("wandb_project", "fitsense-sft"),
        resume="must",
    )
    # Give eval/loss its own x-axis so it can be logged at past training steps
    # without triggering the "step must be monotonically increasing" error.
    wandb.define_metric("eval_step")
    wandb.define_metric("eval/loss", step_metric="eval_step")

    # 4. Evaluate each checkpoint
    n_logged = 0
    for step, ckpt_path in checkpoints:
        logger.info("=" * 60)
        logger.info("Step %d — loading checkpoint: %s", step, ckpt_path)

        try:
            model, tokenizer = load_model_for_eval(str(ckpt_path), config, logger)
        except Exception as exc:
            logger.error("Could not load checkpoint at step %d: %s", step, exc)
            continue

        try:
            val_loss = compute_val_loss_safe(model, tokenizer, val_dataset, logger)
            wandb.log({"eval/loss": val_loss, "eval_step": step})
            logger.info("Logged eval/loss=%.6f at step=%d", val_loss, step)
            n_logged += 1
        except RuntimeError as exc:
            logger.error("Eval failed for step %d: %s", step, exc)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    wandb.finish()
    logger.info(
        "Eval curve complete — %d/%d checkpoints logged.", n_logged, len(checkpoints)
    )


if __name__ == "__main__":
    main()
