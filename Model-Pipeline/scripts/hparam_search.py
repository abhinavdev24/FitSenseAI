"""
Bayesian hyperparameter search for QLoRA SFT training using Optuna.

Each Optuna trial loads a fresh model, injects a LoRA adapter with trial-specific
hyperparameters, trains for a limited number of epochs, and reports validation loss.
The best configuration is saved to `best_hparams.json` in the output directory.

Usage:
    python hparam_search.py --config Model-Pipeline/config/training_config.yaml
    python hparam_search.py --config ... --n-trials 20 --output-dir /tmp/search
    python hparam_search.py --config ... --model-name Qwen/Qwen3-4B --n-trials 5
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from trl import SFTConfig

# ---------------------------------------------------------------------------
# Sibling-module imports (same scripts/ directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_and_validate
from train import (
    build_sft_config,
    extract_best_eval_loss,
    get_git_commit,
    inject_lora_adapter,
    load_config,
    load_model_and_tokenizer,
    run_training,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, Any] = {
    "lora_r": [16, 32, 64],
    "learning_rate_low": 1e-4,
    "learning_rate_high": 5e-4,
    "num_epochs": [2, 3, 5],
    "batch_size": [1, 2, 4],
    "lora_dropout": [0.0, 0.05, 0.1],
    "warmup_ratio": [0.03, 0.05, 0.1],
}


# ---------------------------------------------------------------------------
# Trial logic
# ---------------------------------------------------------------------------

def suggest_hparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest hyperparameter values for a single Optuna trial.

    Args:
        trial: Current Optuna trial instance.

    Returns:
        Dictionary of suggested hyperparameter values.
    """
    lora_r = trial.suggest_categorical("lora_r", SEARCH_SPACE["lora_r"])
    lora_alpha = 2 * lora_r  # derived

    return {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": trial.suggest_categorical(
            "lora_dropout", SEARCH_SPACE["lora_dropout"]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            SEARCH_SPACE["learning_rate_low"],
            SEARCH_SPACE["learning_rate_high"],
            log=True,
        ),
        "num_epochs": trial.suggest_categorical("num_epochs", SEARCH_SPACE["num_epochs"]),
        "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
        "warmup_ratio": trial.suggest_categorical(
            "warmup_ratio", SEARCH_SPACE["warmup_ratio"]
        ),
    }


def build_trial_sft_config(
    config: dict[str, Any],
    trial_dir: Path,
    trial_number: int,
) -> SFTConfig:
    """Build an SFTConfig for a single trial with search-specific overrides.

    Disables W&B reporting and checkpoint saving to keep trials lightweight.

    Args:
        config: Trial-specific config dict (already has hparam overrides applied).
        trial_dir: Temporary output directory for this trial.
        trial_number: Optuna trial number, used to form a unique run name.

    Returns:
        SFTConfig instance configured for a search trial.
    """
    sft_config = build_sft_config(config, trial_dir, f"trial-{trial_number}")
    # Override fields that should differ from a production training run
    sft_config.report_to = "none"
    sft_config.save_strategy = "no"
    sft_config.load_best_model_at_end = False
    return sft_config


def run_trial(
    trial: optuna.Trial,
    base_config: dict[str, Any],
    datasets,
    logger: logging.Logger,
) -> float:
    """Execute a single Optuna trial.

    Loads a fresh model, applies trial hyperparameters, trains, and returns
    the best validation loss. Cleans up temporary files after the trial.

    Args:
        trial: Current Optuna trial instance.
        base_config: Base training config dict (will not be mutated).
        datasets: Pre-loaded DatasetDict with 'train' and 'validation' splits.
        logger: Logger instance.

    Returns:
        Best eval loss achieved during this trial.

    Raises:
        optuna.exceptions.TrialPruned: If the trial is pruned by the pruner.
        ValueError: If eval loss cannot be extracted after training.
    """
    hparams = suggest_hparams(trial)
    logger.info(
        "Trial %d starting — lora_r=%d, lora_alpha=%d, lr=%.2e, "
        "epochs=%d, batch=%d, dropout=%.2f, warmup=%.2f",
        trial.number,
        hparams["lora_r"],
        hparams["lora_alpha"],
        hparams["learning_rate"],
        hparams["num_epochs"],
        hparams["batch_size"],
        hparams["lora_dropout"],
        hparams["warmup_ratio"],
    )

    # Deep-copy base config and apply trial-specific overrides
    trial_config = copy.deepcopy(base_config)
    trial_config.update(hparams)

    trial_dir = Path(tempfile.mkdtemp(prefix=f"hparam_trial_{trial.number}_"))
    try:
        # 1. Fresh model load (required each trial — LoRA config changes)
        model, tokenizer = load_model_and_tokenizer(trial_config, logger)

        # 2. Inject LoRA adapter with trial's lora_r / lora_alpha / lora_dropout
        model = inject_lora_adapter(model, trial_config, logger)

        # 3. Build SFTConfig with trial-specific overrides
        sft_config = build_trial_sft_config(trial_config, trial_dir, trial.number)

        # 4. Train
        trainer = run_training(model, tokenizer, datasets, sft_config, logger)

        # 5. Report intermediate values for pruning (if available)
        for log_entry in trainer.state.log_history:
            if "eval_loss" in log_entry and "step" in log_entry:
                trial.report(log_entry["eval_loss"], step=log_entry["step"])
                if trial.should_prune():
                    logger.info("Trial %d pruned at step %d", trial.number, log_entry["step"])
                    raise optuna.exceptions.TrialPruned()

        # 6. Extract best eval loss
        eval_loss = extract_best_eval_loss(trainer)
        if eval_loss is None:
            raise ValueError(
                f"Trial {trial.number}: could not extract eval loss from trainer state"
            )

        logger.info("Trial %d finished — eval_loss=%.6f", trial.number, eval_loss)
        return eval_loss

    finally:
        # Always clean up temp dir to avoid accumulating stale checkpoints
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir: %s", trial_dir)


def make_objective(
    base_config: dict[str, Any],
    datasets,
    logger: logging.Logger,
):
    """Return a closure that Optuna calls for each trial.

    Args:
        base_config: Base training config dict.
        datasets: Pre-loaded DatasetDict.
        logger: Logger instance.

    Returns:
        Callable that accepts an Optuna Trial and returns eval loss.
    """

    def objective(trial: optuna.Trial) -> float:
        try:
            return run_trial(trial, base_config, datasets, logger)
        except optuna.exceptions.TrialPruned:
            raise  # let Optuna handle pruning
        except Exception as exc:
            logger.error(
                "Trial %d failed with an unexpected error: %s",
                trial.number,
                exc,
                exc_info=True,
            )
            # Return a large sentinel loss so Optuna continues but penalises the trial
            return float("inf")

    return objective


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def build_best_hparams_record(
    study: optuna.Study,
    model_name: str,
) -> dict[str, Any]:
    """Build the best_hparams.json payload from a completed Optuna study.

    Args:
        study: Completed Optuna study.
        model_name: Name of the model used in the search.

    Returns:
        Dictionary ready to be serialised as best_hparams.json.
    """
    best = study.best_trial
    params = dict(best.params)
    # Reconstruct derived lora_alpha (not stored directly by Optuna)
    params["lora_alpha"] = 2 * params["lora_r"]

    return {
        "study_name": study.study_name,
        "model_name": model_name,
        "best_trial_number": best.number,
        "best_eval_loss": best.value,
        "n_trials": len(study.trials),
        "best_params": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_all_trials_record(study: optuna.Study) -> list[dict[str, Any]]:
    """Build the all_trials.json payload from a completed Optuna study.

    Args:
        study: Completed Optuna study.

    Returns:
        List of per-trial summary dicts, sorted by eval loss ascending.
    """
    records = []
    for t in study.trials:
        records.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": dict(t.params),
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                "datetime_complete": (
                    t.datetime_complete.isoformat() if t.datetime_complete else None
                ),
            }
        )
    # Sort by eval loss (None / inf trials go last)
    records.sort(
        key=lambda r: r["value"] if r["value"] is not None else float("inf")
    )
    return records


def save_results(
    output_dir: Path,
    best_record: dict[str, Any],
    all_trials: list[dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Write best_hparams.json and all_trials.json to the output directory.

    Args:
        output_dir: Directory to write result files into.
        best_record: Payload for best_hparams.json.
        all_trials: Payload for all_trials.json.
        logger: Logger instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best_hparams.json"
    with best_path.open("w") as fh:
        json.dump(best_record, fh, indent=2)
    logger.info("Best hyperparameters written to: %s", best_path)

    trials_path = output_dir / "all_trials.json"
    with trials_path.open("w") as fh:
        json.dump(all_trials, fh, indent=2)
    logger.info("All trial results written to: %s", trials_path)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def log_summary_table(
    all_trials: list[dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Log a ranked summary table of all trials to the console.

    Args:
        all_trials: List of trial dicts sorted by eval loss (ascending).
        logger: Logger instance.
    """
    header = (
        f"{'Rank':<5} {'Trial':<7} {'Eval Loss':<12} {'lora_r':<8} "
        f"{'lr':<10} {'epochs':<7} {'batch':<6} {'dropout':<9} {'warmup':<8} {'State'}"
    )
    separator = "-" * len(header)

    logger.info("=" * len(header))
    logger.info("Hyperparameter Search Results — Ranked by Eval Loss")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info(separator)

    for rank, trial in enumerate(all_trials, start=1):
        p = trial.get("params", {})
        loss_str = f"{trial['value']:.6f}" if trial["value"] is not None else "N/A"
        logger.info(
            "%-5d %-7d %-12s %-8s %-10s %-7s %-6s %-9s %-8s %s",
            rank,
            trial["number"],
            loss_str,
            p.get("lora_r", "?"),
            f"{p.get('learning_rate', 0):.2e}",
            p.get("num_epochs", "?"),
            p.get("batch_size", "?"),
            p.get("lora_dropout", "?"),
            p.get("warmup_ratio", "?"),
            trial["state"],
        )

    logger.info("=" * len(header))


def log_wandb_hparam_summary(
    best_record: dict[str, Any],
    all_trials: list[dict[str, Any]],
    study_name: str,
    logger: logging.Logger,
) -> None:
    """Log a single W&B summary run with the best hparam result.

    Logs one W&B run containing best-trial metrics and a summary table of all
    trials. This is intentionally a single run (not one per trial) to avoid
    polluting the W&B workspace.

    Args:
        best_record: Payload from best_hparams.json.
        all_trials: Full trial list for the table artifact.
        study_name: Optuna study name (used as the W&B run name).
        logger: Logger instance.
    """
    try:
        import wandb  # type: ignore[import]

        wandb.init(
        project="fitsense-sft",
        name=f"{study_name}-summary",
        job_type="hparam_search",
        config=best_record.get("best_params", {}),
        )
        wandb.summary["best_eval_loss"] = best_record.get("best_eval_loss")
        wandb.summary["best_trial_number"] = best_record.get("best_trial_number")
        wandb.summary["n_trials"] = best_record.get("n_trials")
        wandb.summary["model_name"] = best_record.get("model_name")

        # Log all-trials as a W&B Table
        columns = ["trial", "eval_loss", "lora_r", "learning_rate", "num_epochs",
                   "batch_size", "lora_dropout", "warmup_ratio", "state"]
        table = wandb.Table(columns=columns)
        for t in all_trials:
            p = t.get("params", {})
            table.add_data(
                t["number"],
                t["value"],
                p.get("lora_r"),
                p.get("learning_rate"),
                p.get("num_epochs"),
                p.get("batch_size"),
                p.get("lora_dropout"),
                p.get("warmup_ratio"),
                t["state"],
            )
        wandb.log({"all_trials": table})
        wandb.finish()
        logger.info("W&B summary run logged for study: %s", study_name)
    except ImportError:
        logger.warning("wandb is not installed — skipping W&B summary logging")
    except Exception as exc:
        logger.warning("Failed to log W&B summary: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the hyperparameter search script."""
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for QLoRA SFT (Optuna + TPE)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Model-Pipeline/config/training_config.yaml",
        help="Path to base training_config.yaml",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials to run (default: 10)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model_name from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Model-Pipeline/outputs/hparam_search",
        help="Directory to save best_hparams.json and all_trials.json",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="fitsense-qlora-hparam-search",
        help="Optuna study name",
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
    """Main entry point for the hyperparameter search script."""
    args = parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger("fitsense.hparam_search", level=log_level)

    # Suppress Optuna's verbose per-trial logging; we handle it ourselves
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. Load base config and apply CLI overrides
    logger.info("Loading base config from: %s", args.config)
    base_config = load_config(args.config)
    if args.model_name is not None:
        base_config["model_name"] = args.model_name
        logger.info("Model name overridden to: %s", args.model_name)

    model_name = base_config["model_name"]
    output_dir = Path(args.output_dir)

    logger.info("Model:       %s", model_name)
    logger.info("Study name:  %s", args.study_name)
    logger.info("N trials:    %d", args.n_trials)
    logger.info("Output dir:  %s", output_dir)
    logger.info("Git commit:  %s", get_git_commit() or "unavailable")

    # 2. Load datasets once — reused across all trials
    train_path = base_config["train_path"]
    val_path = base_config["val_path"]
    logger.info("Loading datasets — train: %s  val: %s", train_path, val_path)
    datasets = load_and_validate(train_path, val_path, logger)

    # 3. Create Optuna study with TPE sampler and MedianPruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # 4. Run the study
    logger.info("Starting Optuna study: %s", args.study_name)
    objective = make_objective(base_config, datasets, logger)
    study.optimize(objective, n_trials=args.n_trials)

    # 5. Check that at least one trial completed successfully
    completed = [t for t in study.trials if t.value is not None and t.value != float("inf")]
    if not completed:
        logger.error(
            "No trials completed successfully. "
            "Check error logs above for details."
        )
        sys.exit(1)

    # 6. Build result records
    best_record = build_best_hparams_record(study, model_name)
    all_trials = build_all_trials_record(study)

    # 7. Log summary table to console
    log_summary_table(all_trials, logger)

    logger.info(
        "Best trial: #%d — eval_loss=%.6f",
        best_record["best_trial_number"],
        best_record["best_eval_loss"],
    )
    logger.info("Best params: %s", json.dumps(best_record["best_params"], indent=2))

    # 8. Save results to disk
    save_results(output_dir, best_record, all_trials, logger)

    # 9. Optionally log final summary to W&B (single run, not per-trial)
    log_wandb_hparam_summary(best_record, all_trials, args.study_name, logger)

    logger.info("Hyperparameter search complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
