"""Load and validate training/validation datasets from JSONL files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

from datasets import DatasetDict, load_dataset


def validate_schema(row: dict[str, Any], row_idx: int, logger: logging.Logger) -> bool:
    """
    Validate a single row against the expected schema.

    Args:
        row: A single row from the dataset
        row_idx: The index of the row (for logging)
        logger: Logger instance

    Returns:
        True if valid, False otherwise
    """
    # Check messages field exists and is a list
    if "messages" not in row:
        logger.warning(f"Row {row_idx}: missing 'messages' field")
        return False

    messages = row.get("messages")
    if not isinstance(messages, list):
        logger.warning(f"Row {row_idx}: 'messages' is not a list")
        return False

    if len(messages) != 3:
        logger.warning(f"Row {row_idx}: expected 3 messages, got {len(messages)}")
        return False

    expected_roles = ["system", "user", "assistant"]
    for i, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
        if not isinstance(msg, dict):
            logger.warning(f"Row {row_idx}: message {i} is not a dict")
            return False

        if "role" not in msg or "content" not in msg:
            logger.warning(
                f"Row {row_idx}: message {i} missing 'role' or 'content' key"
            )
            return False

        if msg["role"] != expected_role:
            logger.warning(
                f"Row {row_idx}: message {i} expected role '{expected_role}', got '{msg['role']}'"
            )
            return False

        if not isinstance(msg["content"], str):
            logger.warning(f"Row {row_idx}: message {i} content is not a string")
            return False

    # Check metadata field
    if "metadata" not in row:
        logger.warning(f"Row {row_idx}: missing 'metadata' field")
        return False

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        logger.warning(f"Row {row_idx}: 'metadata' is not a dict")
        return False

    return True


def compute_stats(
    dataset_dict: DatasetDict, logger: logging.Logger
) -> dict[str, Any]:
    """
    Compute and log statistics about the datasets.

    Args:
        dataset_dict: The loaded dataset dict
        logger: Logger instance

    Returns:
        Dictionary of computed stats
    """
    stats = {}

    for split_name, dataset in dataset_dict.items():
        split_stats: dict[str, Any] = {}
        row_count = len(dataset)
        split_stats["row_count"] = row_count

        thinking_count = 0
        total_chars = 0
        provider_counts: dict[str, int] = {}

        for _row in dataset:
            row = cast(dict[str, Any], _row)
            # Count thinking messages
            messages = row.get("messages", [])
            if len(messages) >= 3:
                assistant_content = messages[2].get("content", "")
                if assistant_content.strip().startswith("<think>"):
                    thinking_count += 1

            # Count characters
            for msg in messages:
                content = msg.get("content", "")
                total_chars += len(content)

            # Count providers
            metadata = row.get("metadata", {})
            provider = metadata.get("provider", "unknown")
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        split_stats["thinking_count"] = thinking_count
        split_stats["non_thinking_count"] = row_count - thinking_count
        split_stats["thinking_ratio"] = (
            thinking_count / row_count if row_count > 0 else 0
        )
        split_stats["approx_tokens"] = total_chars // 4
        split_stats["provider_breakdown"] = provider_counts

        stats[split_name] = split_stats

    # Log stats
    logger.info("=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)

    for split_name, split_stats in stats.items():
        logger.info(f"\n{split_name.upper()} SPLIT:")
        logger.info(f"  Rows: {split_stats['row_count']}")
        logger.info(
            f"  Reasoning: {split_stats['thinking_count']} rows "
            f"({split_stats['thinking_ratio']:.1%})"
        )
        logger.info(f"  Approx tokens: {split_stats['approx_tokens']:,}")
        logger.info("  Provider breakdown:")
        for provider, count in split_stats["provider_breakdown"].items():
            pct = count / split_stats["row_count"] * 100
            logger.info(f"    - {provider}: {count} ({pct:.1f}%)")

    logger.info("=" * 60)

    return stats


def load_and_validate(
    train_path: str,
    val_path: str,
    logger: logging.Logger | None = None,
) -> DatasetDict:
    """
    Load train and validation datasets, validate schema, and compute stats.

    Args:
        train_path: Path to train.jsonl
        val_path: Path to validation.jsonl
        logger: Optional logger instance (will be created if not provided)

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Convert paths to strings if they're Path objects
    train_path = str(train_path)
    val_path = str(val_path)

    # Verify files exist
    train_p = Path(train_path)
    val_p = Path(val_path)

    if not train_p.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_p.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    logger.info(f"Loading train from: {train_path}")
    logger.info(f"Loading validation from: {val_path}")

    # Load datasets
    dataset_dict = cast(
        DatasetDict,
        load_dataset(
            "json",
            data_files={"train": train_path, "validation": val_path},
        ),
    )

    # Validate schema for each row
    invalid_rows = {"train": 0, "validation": 0}

    for split_name in ["train", "validation"]:
        dataset = dataset_dict[split_name]
        for row_idx, row in enumerate(dataset):
            if not validate_schema(cast(dict[str, Any], row), row_idx, logger):
                invalid_rows[split_name] += 1

    # Log validation results
    logger.info("Schema Validation Results:")
    for split_name, invalid_count in invalid_rows.items():
        total = len(dataset_dict[split_name])
        valid_count = total - invalid_count
        logger.info(
            f"  {split_name}: {valid_count}/{total} valid "
            f"({invalid_count} warnings)"
        )

    # Compute and log stats
    compute_stats(dataset_dict, logger)

    return dataset_dict


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Load and validate training/validation datasets"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        required=True,
        help="Path to val.jsonl",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to write logs",
    )

    args = parser.parse_args()

    # Setup logger
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "load_data.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    # Load and validate
    try:
        load_and_validate(args.train_path, args.val_path, logger)
        logger.info("Dataset loading and validation completed successfully")
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
