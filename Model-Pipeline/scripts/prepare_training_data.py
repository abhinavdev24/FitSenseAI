"""
Prepare teacher LLM responses for Qwen3 SFT with thinking distillation.

Reads the re-reasoned responses.jsonl (all OpenRouter, all have reasoning)
and produces train.jsonl / val.jsonl in the Hugging Face "messages" format
that SFTTrainer + Qwen3 chat template expects.

Output format per line (JSONL):
  {"messages": [
      {"role": "system", "content": "…"},
      {"role": "user",   "content": "…"},
      {"role": "assistant", "content": "<think>\\n…\\n</think>\\n{json}"}
  ]}

Usage:
    conda activate mlopsenv
    python Model-Pipeline/prepare_training_data.py \\
        --input  Data-Pipeline/data/raw/teacher-llm-responses/20260324T162857Z/responses.jsonl \\
        --output Model-Pipeline/data/training
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validate a single record
# ---------------------------------------------------------------------------

def _validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of issues (empty == valid)."""
    issues: list[str] = []

    if record.get("status") != "success":
        issues.append(f"status={record.get('status')}")

    messages = (record.get("request_payload") or {}).get("messages", [])
    if len(messages) < 2:
        issues.append(f"request has only {len(messages)} message(s)")

    if not (record.get("response_text") or "").strip():
        issues.append("empty response_text")

    if record.get("response_json") is None:
        issues.append("response_json is null")

    raw = record.get("raw_response") or {}
    choices = raw.get("choices") or [{}]
    reasoning = (choices[0].get("message") or {}).get("reasoning")
    if not reasoning:
        issues.append("missing reasoning")

    return issues


# ---------------------------------------------------------------------------
# Convert one raw record → training sample
# ---------------------------------------------------------------------------

def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a raw teacher record into a messages-format training sample.

    Returns None if the record is invalid.
    """
    issues = _validate_record(record)
    if issues:
        logger.warning("Skipping %s: %s", record.get("response_id", "?"), "; ".join(issues))
        return None

    raw = record["raw_response"]
    reasoning: str = raw["choices"][0]["message"]["reasoning"].strip()
    answer: str = record["response_text"].strip()
    # Strip any residual <think> block from answer (shouldn't exist, but be safe)
    answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL).strip()

    assistant_content = f"<think>\n{reasoning}\n</think>\n{answer}"

    payload_messages = record["request_payload"]["messages"]
    return {
        "messages": [
            {"role": "system",    "content": payload_messages[0]["content"]},
            {"role": "user",      "content": payload_messages[1]["content"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "response_id": record.get("response_id"),
            "query_id":    record.get("query_id"),
            "prompt_type": record.get("prompt_type"),
            "model_name":  record.get("model_name"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare teacher LLM responses for Qwen3 SFT training."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to raw responses.jsonl from the teacher pipeline.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory. Will write train.jsonl and val.jsonl here.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of data reserved for validation (default 0.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default 42).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & convert ----
    samples: list[dict[str, Any]] = []
    skipped = 0
    line_num = 0

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            sample = convert_record(record)
            if sample is None:
                skipped += 1
            else:
                samples.append(sample)

    logger.info("Loaded %d records, converted %d, skipped %d", line_num, len(samples), skipped)

    # ---- Train / val split ----
    random.seed(args.seed)
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    logger.info("Split: %d train, %d val", len(train_samples), len(val_samples))

    # ---- Write output ----
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    for path, data in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w") as f:
            for sample in data:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")

    # ---- Summary ----
    def _token_estimate(samples: list[dict]) -> int:
        return sum(len(m["content"]) for s in samples for m in s["messages"]) // 4

    summary = {
        "input_file":          str(input_path),
        "total_raw_records":   line_num,
        "converted":           len(samples),
        "skipped":             skipped,
        "train_count":         len(train_samples),
        "val_count":           len(val_samples),
        "train_approx_tokens": _token_estimate(train_samples),
        "val_approx_tokens":   _token_estimate(val_samples),
        "val_ratio":           args.val_ratio,
        "seed":                args.seed,
    }

    summary_path = output_dir / "prepare_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote %s", train_path)
    logger.info("Wrote %s", val_path)
    logger.info("Wrote %s", summary_path)
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
