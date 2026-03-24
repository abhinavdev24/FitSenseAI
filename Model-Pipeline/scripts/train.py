"""
QLoRA-based SFT training for FitSenseAI student model (Qwen3).

Trains a LoRA adapter on top of a 4-bit quantised base model using Unsloth
for fast kernel support and TRL SFTTrainer for loss-masked fine-tuning.
Training data is multi-turn tool-calling conversations with <think> reasoning
traces produced by the teacher-distillation pipeline.

Usage:
    python train.py --config Model-Pipeline/config/training_config.yaml
    python train.py --config ... --model-name Qwen/Qwen3-4B
    python train.py --config ... --output-dir /tmp/run1
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
import yaml
from trl import SFTConfig, SFTTrainer

# load_data is a sibling module in the same scripts/ directory
sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_and_validate


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to stdout with a standard format."""
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
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load and return training config from a YAML file.

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


def apply_cli_overrides(
    config: dict[str, Any],
    model_name: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    """Override config values with CLI arguments when provided.

    Args:
        config: Base config dict loaded from YAML.
        model_name: Optional CLI model name override.
        output_dir: Optional CLI output directory override.

    Returns:
        Updated config dict.
    """
    if model_name is not None:
        config["model_name"] = model_name
    if output_dir is not None:
        config["output_dir"] = output_dir
    return config


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

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
# Model helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: dict[str, Any], logger: logging.Logger):
    """Load the base model and tokenizer via Unsloth FastLanguageModel.

    Uses 4-bit NF4 quantisation (handled internally by Unsloth).

    Args:
        config: Training config dictionary.
        logger: Logger instance.

    Returns:
        Tuple of (model, tokenizer).

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

    logger.info("Loading model: %s (4-bit)", config["model_name"])
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_name"],
            max_seq_length=config["max_seq_length"],
            load_in_4bit=True,
            dtype=None,  # auto-detect
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{config['model_name']}': {exc}"
        ) from exc

    logger.info("Model loaded successfully")
    return model, tokenizer


def inject_lora_adapter(model, config: dict[str, Any], logger: logging.Logger):
    """Inject LoRA adapter into the base model via Unsloth.

    Args:
        model: The loaded base model.
        config: Training config dictionary.
        logger: Logger instance.

    Returns:
        Model with LoRA adapter injected.

    Raises:
        ImportError: If unsloth is not installed.
        RuntimeError: If adapter injection fails.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("unsloth is required") from exc

    logger.info(
        "Injecting LoRA adapter — r=%d, alpha=%d, dropout=%.2f",
        config["lora_r"],
        config["lora_alpha"],
        config["lora_dropout"],
    )
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except Exception as exc:
        raise RuntimeError(f"LoRA adapter injection failed: {exc}") from exc

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_sft_config(
    config: dict[str, Any],
    output_dir: Path,
    run_name: str,
) -> SFTConfig:
    """Build TRL SFTConfig from the training config dictionary.

    Args:
        config: Training config dictionary.
        output_dir: Directory for checkpoints and logs.
        run_name: W&B run name.

    Returns:
        Configured SFTConfig instance.
    """
    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        bf16=config.get("bf16", True),
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=config["max_seq_length"],
        packing=True,
        dataset_text_field="messages",
        report_to=config.get("report_to", "wandb"),
        run_name=run_name,
    )


def run_training(
    model,
    tokenizer,
    datasets,
    sft_config: SFTConfig,
    logger: logging.Logger,
) -> SFTTrainer:
    """Initialise and run the SFTTrainer.

    Args:
        model: LoRA-enabled model.
        tokenizer: Tokenizer matching the model.
        datasets: DatasetDict with 'train' and 'validation' splits.
        sft_config: TRL SFTConfig.
        logger: Logger instance.

    Returns:
        Trainer instance after training completes.

    Raises:
        RuntimeError: If training fails.
    """
    logger.info("Initialising SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=sft_config,
    )

    logger.info("Starting training")
    try:
        trainer.train()
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc

    logger.info("Training complete")
    return trainer


# ---------------------------------------------------------------------------
# Post-training
# ---------------------------------------------------------------------------

def save_adapter_and_tokenizer(
    model,
    tokenizer,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save the final LoRA adapter and tokenizer to disk.

    Args:
        model: Trained model with LoRA adapter.
        tokenizer: Matching tokenizer.
        output_dir: Parent output directory.
        logger: Logger instance.
    """
    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving LoRA adapter to: %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Adapter and tokenizer saved")


def extract_best_eval_loss(trainer: SFTTrainer) -> float | None:
    """Extract the best evaluation loss recorded during training.

    Args:
        trainer: Completed SFTTrainer instance.

    Returns:
        Best eval loss as a float, or None if not available.
    """
    state = trainer.state
    if state is None:
        return None
    return getattr(state, "best_metric", None)


def write_training_summary(
    output_dir: Path,
    config: dict[str, Any],
    trainer: SFTTrainer,
    run_name: str,
    elapsed_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Write training_summary.json to the output directory.

    Args:
        output_dir: Parent output directory.
        config: Training config used for this run.
        trainer: Completed SFTTrainer instance.
        run_name: W&B run name.
        elapsed_seconds: Wall-clock training time in seconds.
        logger: Logger instance.

    Returns:
        Summary dictionary that was written to disk.
    """
    git_sha = get_git_commit()
    best_eval_loss = extract_best_eval_loss(trainer)
    total_steps = trainer.state.global_step if trainer.state else None

    summary: dict[str, Any] = {
        "model_name": config["model_name"],
        "run_name": run_name,
        "total_steps": total_steps,
        "best_eval_loss": best_eval_loss,
        "training_time_seconds": round(elapsed_seconds, 1),
        "training_time_human": _format_duration(elapsed_seconds),
        "git_commit": git_sha,
        "config": config,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Training summary written to: %s", summary_path)
    return summary


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def log_wandb_summary(summary: dict[str, Any], output_dir: Path, logger: logging.Logger) -> None:
    """Log the training summary and final adapter as a W&B artifact.

    Args:
        summary: Training summary dictionary.
        output_dir: Parent output directory (contains final_adapter/).
        logger: Logger instance.
    """
    try:
        # Scalar summary metrics
        wandb.summary["best_eval_loss"] = summary.get("best_eval_loss")
        wandb.summary["total_steps"] = summary.get("total_steps")
        wandb.summary["training_time_seconds"] = summary.get("training_time_seconds")
        wandb.summary["git_commit"] = summary.get("git_commit")

        # Model artifact
        artifact = wandb.Artifact(
            name="fitsense-qlora-adapter",
            type="model",
            description=f"QLoRA adapter for {summary['model_name']}",
            metadata={"model_name": summary["model_name"], "config": summary["config"]},
        )
        adapter_dir = output_dir / "final_adapter"
        if adapter_dir.exists():
            artifact.add_dir(str(adapter_dir), name="final_adapter")
        summary_path = output_dir / "training_summary.json"
        if summary_path.exists():
            artifact.add_file(str(summary_path))

        wandb.log_artifact(artifact)
        logger.info("W&B artifact logged: fitsense-qlora-adapter")
    except Exception as exc:
        logger.warning("Failed to log W&B artifact: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QLoRA SFT training for FitSenseAI (Qwen3 + Unsloth + TRL)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Model-Pipeline/config/training_config.yaml",
        help="Path to training_config.yaml",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model_name from config (e.g. Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from config",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main training entrypoint."""
    args = parse_args()
    logger = setup_logger("fitsense.train", level=getattr(logging, args.log_level))

    # 1. Load and apply config
    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)
    config = apply_cli_overrides(config, args.model_name, args.output_dir)

    model_name_short = config["model_name"].split("/")[-1].lower()
    run_name = f"fitsense-{model_name_short}-qlora"
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model:      %s", config["model_name"])
    logger.info("Output dir: %s", output_dir)
    logger.info("Run name:   %s", run_name)

    # 2. Initialise W&B
    logger.info("Initialising W&B run")
    wandb.init(
        project=config.get("wandb_project", "fitsense-sft"),
        name=run_name,
        config=config,
    )

    # 3. Load and validate training data
    train_path = config["train_path"]
    val_path = config["val_path"]
    logger.info("Loading datasets — train: %s  val: %s", train_path, val_path)
    datasets = load_and_validate(train_path, val_path, logger)

    # 4. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, logger)

    # 5. Inject LoRA adapter
    model = inject_lora_adapter(model, config, logger)

    # 6. Build SFTConfig
    sft_config = build_sft_config(config, output_dir, run_name)

    # 7. Train
    start_time = time.monotonic()
    trainer = run_training(model, tokenizer, datasets, sft_config, logger)
    elapsed = time.monotonic() - start_time
    logger.info("Total training wall time: %s", _format_duration(elapsed))

    # 8. Save final adapter
    save_adapter_and_tokenizer(model, tokenizer, output_dir, logger)

    # 9. Write training summary
    summary = write_training_summary(
        output_dir, config, trainer, run_name, elapsed, logger
    )

    # 10. Log summary and model artifact to W&B
    log_wandb_summary(summary, output_dir, logger)

    wandb.finish()
    logger.info("Training pipeline complete. Outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
