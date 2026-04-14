"""Merge LoRA adapter into base model and produce bf16 + AWQ-int4 artifacts.

Loads a trained LoRA adapter, merges it into the full-precision base model,
saves the merged weights as bf16, then optionally quantizes to AWQ int4 using
calibration data drawn from the training JSONL.

Usage:
    python merge_weights.py --config Model-Pipeline/config/training_config.yaml
    python merge_weights.py --adapter-dir Model-Pipeline/outputs/final_adapter
    python merge_weights.py --skip-awq --output-dir /tmp/merged

The base model is always read from model_name in the config file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to stdout with a standard format.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO).

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


def resolve_adapter_dir(
    config: dict[str, Any],
    adapter_dir_override: str | None,
    logger: logging.Logger,
) -> Path:
    """Resolve the LoRA adapter directory from config or CLI override.

    The adapter dir is: config['output_dir'] + '/final_adapter', unless
    overridden by --adapter-dir on the CLI.

    Args:
        config: Training config dictionary.
        adapter_dir_override: Optional CLI override for adapter directory.
        logger: Logger instance.

    Returns:
        Resolved Path to the adapter directory.

    Raises:
        FileNotFoundError: If the resolved adapter directory does not exist.
    """
    if adapter_dir_override is not None:
        adapter_dir = Path(adapter_dir_override)
        logger.info("Adapter dir (CLI override): %s", adapter_dir)
    else:
        output_dir = config.get("output_dir", "Model-Pipeline/outputs")
        adapter_dir = Path(output_dir) / "final_adapter"
        logger.info("Adapter dir (from config): %s", adapter_dir)

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory does not exist: {adapter_dir}. "
            "Run train.py first to produce the adapter."
        )
    return adapter_dir


# ---------------------------------------------------------------------------
# Step 1: Merge LoRA → bf16
# ---------------------------------------------------------------------------


def merge_lora_to_bf16(
    adapter_dir: Path,
    bf16_output_dir: Path,
    max_seq_length: int,
    logger: logging.Logger,
) -> None:
    """Merge a LoRA adapter into the base model and save as bf16.

    Uses unsloth's FastLanguageModel to load the adapter (which resolves the
    base model from adapter_config.json), then exports merged bf16 weights via
    save_pretrained_merged. This correctly dequantizes 4-bit training models.

    Args:
        adapter_dir: Path to the saved LoRA adapter directory.
        bf16_output_dir: Output directory for the merged bf16 model.
        max_seq_length: Max sequence length passed to FastLanguageModel.
        logger: Logger instance.

    Raises:
        ImportError: If unsloth is not installed.
        RuntimeError: If model loading or saving fails.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "unsloth is required for merging. Install with: pip install unsloth"
        ) from exc

    bf16_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model + adapter via unsloth from: %s", adapter_dir)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load model via unsloth: {exc}") from exc

    logger.info("Saving merged bf16 model to: %s", bf16_output_dir)
    try:
        model.save_pretrained_merged(
            str(bf16_output_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save merged model to '{bf16_output_dir}': {exc}"
        ) from exc

    logger.info("bf16 merge complete — output: %s", bf16_output_dir)


# ---------------------------------------------------------------------------
# Step 2: Load calibration data
# ---------------------------------------------------------------------------


def _messages_to_plain_text(messages: list[dict[str, Any]]) -> str:
    """Format a messages list as plain text when chat template is unavailable.

    Falls back to joining each message as 'role: content' pairs separated
    by newlines.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Flat string representation of the conversation.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # content may be a list of content blocks (tool call format)
        if isinstance(content, list):
            content_str = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            content_str = str(content)
        parts.append(f"{role}: {content_str}")
    return "\n".join(parts)


def load_calibration_data(
    calibration_data_path: str,
    num_samples: int,
    tokenizer: Any,
    logger: logging.Logger,
) -> list[str]:
    """Load and format calibration texts from a training JSONL file.

    Reads up to num_samples rows from the JSONL, applies the tokenizer's
    apply_chat_template to produce flat text strings. Falls back to plain-text
    formatting if the chat template raises (e.g. missing system role).

    Args:
        calibration_data_path: Path to the training JSONL file.
        num_samples: Maximum number of samples to load.
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        logger: Logger instance.

    Returns:
        List of formatted text strings for AWQ calibration.

    Raises:
        FileNotFoundError: If the calibration data file does not exist.
        ValueError: If no valid samples could be loaded.
    """
    import json

    path = Path(calibration_data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration data not found: {calibration_data_path}"
        )

    calibration_texts: list[str] = []
    fallback_count = 0
    skipped_count = 0

    logger.info(
        "Loading calibration data from: %s (max %d samples)",
        path,
        num_samples,
    )

    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            if len(calibration_texts) >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d — invalid JSON: %s", line_num, exc)
                skipped_count += 1
                continue

            messages = record.get("messages")
            if not messages or not isinstance(messages, list):
                logger.warning(
                    "Skipping line %d — missing or invalid 'messages' field",
                    line_num,
                )
                skipped_count += 1
                continue

            # Try apply_chat_template; fall back to plain text on any error
            try:
                text: str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                logger.debug(
                    "Line %d: apply_chat_template failed (%s) — using plain-text fallback",
                    line_num,
                    exc,
                )
                text = _messages_to_plain_text(messages)
                fallback_count += 1

            calibration_texts.append(text)

    if not calibration_texts:
        raise ValueError(
            f"No valid calibration samples loaded from {calibration_data_path}. "
            f"Skipped {skipped_count} lines."
        )

    logger.info(
        "Loaded %d calibration samples (fallback plain-text: %d, skipped: %d)",
        len(calibration_texts),
        fallback_count,
        skipped_count,
    )
    return calibration_texts


# ---------------------------------------------------------------------------
# Step 3: GPTQ quantization via llm-compressor
# ---------------------------------------------------------------------------


def quantize_awq(
    bf16_output_dir: Path,
    awq_output_dir: Path,
    calibration_texts: list[str],
    awq_bits: int,
    awq_group_size: int,
    max_seq_length: int,
    logger: logging.Logger,
) -> None:
    """Quantize a bf16 model using GPTQ via llm-compressor.

    Uses llmcompressor's oneshot API with GPTQModifier to apply weight-only
    quantization calibrated on training data.

    Args:
        bf16_output_dir: Path to the merged bf16 model directory.
        awq_output_dir: Output directory for the quantized model.
        calibration_texts: List of calibration text samples.
        awq_bits: Bit-width for quantization (4 or 8).
        awq_group_size: Group size for quantization (typically 128).
        max_seq_length: Maximum sequence length (must match training config).
        logger: Logger instance.

    Raises:
        ImportError: If llmcompressor is not installed.
        RuntimeError: If any quantization step fails.
    """
    try:
        from datasets import Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as exc:
        raise ImportError(
            "llmcompressor and datasets are required for GPTQ quantization."
        ) from exc

    bits_to_scheme = {4: "W4A16", 8: "W8A16"}
    if awq_bits not in bits_to_scheme:
        raise ValueError(f"Unsupported awq_bits={awq_bits}; expected 4 or 8.")

    scheme = bits_to_scheme[awq_bits]
    logger.info(
        "Quantization config — scheme=%s, group_size=%d, max_seq_length=%d",
        scheme,
        awq_group_size,
        max_seq_length,
    )

    calib_dataset = Dataset.from_dict({"text": calibration_texts})

    recipe = GPTQModifier(
        targets="Linear",
        scheme=scheme,
        block_size=awq_group_size,
        ignore=["lm_head"],
    )

    logger.info("Running GPTQ oneshot on model from: %s", bf16_output_dir)
    try:
        oneshot(
            model=str(bf16_output_dir),
            dataset=calib_dataset,
            recipe=recipe,
            output_dir=str(awq_output_dir),
            max_seq_length=max_seq_length,
            num_calibration_samples=len(calibration_texts),
        )
    except Exception as exc:
        raise RuntimeError(
            f"GPTQ quantization failed for '{bf16_output_dir}': {exc}"
        ) from exc

    logger.info("Quantization complete — output: %s", awq_output_dir)


# ---------------------------------------------------------------------------
# Step 2: Inference check
# ---------------------------------------------------------------------------


def run_inference_check(
    bf16_output_dir: Path,
    train_data_path: str,
    num_samples: int,
    max_new_tokens: int,
    max_seq_length: int,
    logger: logging.Logger,
) -> None:
    """Run greedy inference on a few training samples to verify the merged model.

    For each sample, all messages before the final assistant turn are used as
    the prompt (with add_generation_prompt=True). The model's response is logged
    alongside a truncated view of the last user message.

    Args:
        bf16_output_dir: Path to the merged bf16 model directory.
        train_data_path: Path to the training JSONL file.
        num_samples: Number of samples to run inference on.
        max_new_tokens: Maximum tokens to generate per sample.
        max_seq_length: Max sequence length passed to FastLanguageModel.
        logger: Logger instance.
    """
    import json

    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "unsloth and torch are required for inference check."
        ) from exc

    path = Path(train_data_path)
    if not path.exists():
        logger.warning("Train data not found at %s — skipping inference check", path)
        return

    samples: list[list[dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = record.get("messages")
            if messages and isinstance(messages, list):
                samples.append(messages)
            if len(samples) >= num_samples:
                break

    if not samples:
        logger.warning("No valid samples found — skipping inference check")
        return

    logger.info(
        "Loading merged model for inference check from: %s", bf16_output_dir
    )
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(bf16_output_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(model)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load merged model for inference: {exc}"
        ) from exc

    logger.info("Running inference on %d sample(s)", len(samples))
    for i, messages in enumerate(samples, start=1):
        # Prompt = all messages before the last assistant turn
        last_assistant = next(
            (j for j in range(len(messages) - 1, -1, -1)
             if messages[j].get("role") == "assistant"),
            None,
        )
        prompt_messages = messages[:last_assistant] if last_assistant else messages[:1]

        try:
            prompt_text: str = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = _messages_to_plain_text(prompt_messages)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)

        last_user = next(
            (m["content"] for m in reversed(prompt_messages)
             if m.get("role") == "user"),
            "<no user message>",
        )
        if isinstance(last_user, list):
            last_user = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in last_user
            )

        logger.info("=== Inference sample %d/%d ===", i, len(samples))
        logger.info("PROMPT (last user turn): %.300s", last_user)
        logger.info("RESPONSE:                %.500s", response)

    logger.info("Inference check complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge a LoRA adapter into the base model (bf16) and optionally "
            "produce an AWQ int4 quantized artifact."
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
        default=None,
        help=(
            "Path to the LoRA adapter directory. Overrides the path derived from "
            "config output_dir + '/final_adapter'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Model-Pipeline/outputs/final_merged",
        help="Root output directory for bf16 and AWQ artifacts (default: Model-Pipeline/outputs/final_merged)",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default="Model-Pipeline/data/training/train.jsonl",
        help=(
            "Path to training JSONL for AWQ calibration. "
            "Each line must have a 'messages' key. "
            "(default: Model-Pipeline/data/training/train.jsonl)"
        ),
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=128,
        help="Number of calibration samples to use for AWQ (default: 128)",
    )
    parser.add_argument(
        "--awq-bits",
        type=int,
        default=4,
        help="Bit-width for AWQ quantization (default: 4)",
    )
    parser.add_argument(
        "--awq-group-size",
        type=int,
        default=128,
        help="Group size for AWQ quantization (default: 128)",
    )
    parser.add_argument(
        "--skip-awq",
        action="store_true",
        help="Skip the AWQ quantization step; only produce the bf16 artifact.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip LoRA merge + inference check; use existing bf16 model for quantization.",
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
    """Main entrypoint: merge LoRA adapter and produce bf16 + AWQ artifacts."""
    args = parse_args()
    logger = setup_logger(
        "fitsense.merge_weights", level=getattr(logging, args.log_level)
    )

    # 1. Load config and resolve adapter directory
    logger.info("Loading config from: %s", args.config)
    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        logger.error("Config load failed: %s", exc)
        sys.exit(1)

    try:
        adapter_dir = resolve_adapter_dir(config, args.adapter_dir, logger)
    except FileNotFoundError as exc:
        logger.error("Adapter directory resolution failed: %s", exc)
        sys.exit(1)

    base_model = config.get("model_name")
    if not base_model:
        logger.error("model_name not set in config: %s", args.config)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    bf16_output_dir = output_dir / "bf16"
    awq_output_dir = output_dir / "awq"

    logger.info("Base model:       %s", base_model)
    logger.info("Adapter dir:      %s", adapter_dir)
    logger.info("bf16 output dir:  %s", bf16_output_dir)
    if not args.skip_awq:
        logger.info("AWQ output dir:   %s", awq_output_dir)
    logger.info("Skip AWQ:         %s", args.skip_awq)

    max_seq_length: int = config.get("max_seq_length", 2048)

    # 2. Merge LoRA adapter into base model and save as bf16
    if args.skip_merge:
        if not bf16_output_dir.exists():
            logger.error(
                "--skip-merge set but bf16 dir does not exist: %s", bf16_output_dir
            )
            sys.exit(1)
        logger.info("--skip-merge set — using existing bf16 model at %s", bf16_output_dir)
    else:
        logger.info("--- Step 1/4: Merging LoRA adapter into bf16 base model ---")
        try:
            merge_lora_to_bf16(
                adapter_dir=adapter_dir,
                bf16_output_dir=bf16_output_dir,
                max_seq_length=max_seq_length,
                logger=logger,
            )
        except (ImportError, RuntimeError) as exc:
            logger.error("bf16 merge step failed: %s", exc)
            sys.exit(1)

        # 3. Inference check on merged bf16 model
        logger.info("--- Step 2/4: Inference check on merged model ---")
        try:
            run_inference_check(
                bf16_output_dir=bf16_output_dir,
                train_data_path=config.get("train_path", args.calibration_data),
                num_samples=3,
                max_new_tokens=256,
                max_seq_length=max_seq_length,
                logger=logger,
            )
        except (ImportError, RuntimeError) as exc:
            logger.warning("Inference check failed (non-fatal): %s", exc)

    if args.skip_awq:
        logger.info("--skip-awq set — skipping AWQ quantization step")
        logger.info("Done. bf16 model saved to: %s", bf16_output_dir)
        return

    # 4. Load calibration data (requires the tokenizer from the bf16 output)
    logger.info("--- Step 3/4: Loading calibration data ---")
    try:
        from transformers import AutoTokenizer

        calib_tokenizer = AutoTokenizer.from_pretrained(str(bf16_output_dir))
    except Exception as exc:
        logger.error(
            "Failed to load tokenizer from bf16 output for calibration: %s", exc
        )
        sys.exit(1)

    try:
        calibration_texts = load_calibration_data(
            calibration_data_path=args.calibration_data,
            num_samples=args.num_calibration_samples,
            tokenizer=calib_tokenizer,
            logger=logger,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Calibration data load failed: %s", exc)
        sys.exit(1)

    # 5. INT4 quantization via llm-compressor
    logger.info("--- Step 4/4: INT%d quantization (llm-compressor/GPTQ) ---", args.awq_bits)
    try:
        quantize_awq(
            bf16_output_dir=bf16_output_dir,
            awq_output_dir=awq_output_dir,
            calibration_texts=calibration_texts,
            awq_bits=args.awq_bits,
            awq_group_size=args.awq_group_size,
            max_seq_length=max_seq_length,
            logger=logger,
        )
    except (ImportError, RuntimeError) as exc:
        logger.error("AWQ quantization step failed: %s", exc)
        sys.exit(1)

    logger.info("merge_weights complete.")
    logger.info("  bf16 model : %s", bf16_output_dir)
    logger.info("  AWQ model  : %s", awq_output_dir)


if __name__ == "__main__":
    main()
