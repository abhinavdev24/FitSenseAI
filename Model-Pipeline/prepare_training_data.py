"""
Prepare teacher LLM responses for Qwen3 SFT with thinking distillation.

Reads the raw responses.jsonl from the teacher pipeline and produces
train.jsonl / val.jsonl in the Hugging Face "messages" format that
SFTTrainer + Qwen3 chat template expects.

Three source record types are handled:
  1. Groq  — reasoning is inline as <think>…</think> in response_text
  2. OpenRouter (with reasoning) — reasoning in raw_response.choices[0].message.reasoning,
     content is the final JSON answer
  3. OpenRouter (no reasoning) — content only, no thinking trace

Output format per line (JSONL):
  {"messages": [
      {"role": "system", "content": "…"},
      {"role": "user",   "content": "…"},
      {"role": "assistant", "content": "<think>\\n…\\n</think>\\n{json}"}
  ]}

Records without reasoning (type 3) are kept — the assistant content is
just the JSON with no <think> block, so the model also learns the
"non-thinking" path.

Usage:
    conda activate mlopsenv
    python Model-Pipeline/scripts/prepare_training_data.py \
        --input  Data-Pipeline/data/raw/teacher-llm-responses/20260308T234052Z/responses.jsonl \
        --output Model-Pipeline/data/training \
        --val-ratio 0.1 \
        --seed 42
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
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_groq(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (reasoning, answer) from a Groq record.

    Groq embeds <think>…</think> at the start of response_text followed by
    the JSON answer.  We split on the closing tag.
    """
    text: str = record.get("response_text", "")
    match = re.match(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # Fallback: no think block found — treat whole text as answer
    return None, text.strip()


def _extract_openrouter(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (reasoning, answer) from an OpenRouter record.

    Reasoning lives in raw_response.choices[0].message.reasoning (may be None).
    The answer is in response_text (== raw_response.choices[0].message.content).
    """
    raw = record.get("raw_response", {})
    choices = raw.get("choices", [])
    reasoning = None
    if choices:
        msg = choices[0].get("message", {})
        reasoning = msg.get("reasoning") or None
    if reasoning:
        reasoning = reasoning.strip()

    answer = record.get("response_text", "").strip()
    return reasoning, answer


# ---------------------------------------------------------------------------
# Build the assistant content
# ---------------------------------------------------------------------------

def _build_assistant_content(reasoning: str | None, answer: str) -> str:
    """Wrap reasoning in <think> tags and append the answer."""
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n{answer}"
    return answer


# ---------------------------------------------------------------------------
# Validate a single record
# ---------------------------------------------------------------------------

def _validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of issues (empty == valid)."""
    issues: list[str] = []

    if record.get("status") != "success":
        issues.append(f"status={record.get('status')}")

    payload = record.get("request_payload", {})
    messages = payload.get("messages", [])
    if len(messages) < 2:
        issues.append(f"request has only {len(messages)} message(s)")

    response_text = record.get("response_text", "")
    if not response_text.strip():
        issues.append("empty response_text")

    # The response should contain valid JSON (the workout plan)
    response_json = record.get("response_json")
    if response_json is None:
        issues.append("response_json is null")

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
        logger.warning(
            "Skipping %s: %s",
            record.get("response_id", "?"),
            "; ".join(issues),
        )
        return None

    provider = record.get("provider", "")
    if provider == "groq":
        reasoning, answer = _extract_groq(record)
    else:
        reasoning, answer = _extract_openrouter(record)

    if not answer:
        logger.warning(
            "Skipping %s: empty answer after extraction", record.get("response_id")
        )
        return None

    assistant_content = _build_assistant_content(reasoning, answer)

    # Pull original system + user messages from the request payload
    payload_messages = record["request_payload"]["messages"]
    system_msg = payload_messages[0]  # {"role": "system", "content": "…"}
    user_msg = payload_messages[1]    # {"role": "user",   "content": "…"}

    return {
        "messages": [
            {"role": "system", "content": system_msg["content"]},
            {"role": "user", "content": user_msg["content"]},
            {"role": "assistant", "content": assistant_content},
        ],
        # Metadata for traceability (not used by SFTTrainer)
        "metadata": {
            "response_id": record.get("response_id"),
            "query_id": record.get("query_id"),
            "prompt_type": record.get("prompt_type"),
            "provider": provider,
            "model_name": record.get("model_name"),
            "has_reasoning": reasoning is not None,
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
    stats = {
        "groq_with_thinking": 0,
        "openrouter_with_reasoning": 0,
        "openrouter_no_reasoning": 0,
    }

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            sample = convert_record(record)
            if sample is None:
                skipped += 1
                continue

            # Track stats
            meta = sample["metadata"]
            if meta["provider"] == "groq" and meta["has_reasoning"]:
                stats["groq_with_thinking"] += 1
            elif meta["provider"] == "openrouter" and meta["has_reasoning"]:
                stats["openrouter_with_reasoning"] += 1
            else:
                stats["openrouter_no_reasoning"] += 1

            samples.append(sample)

    logger.info("Loaded %d records, converted %d, skipped %d", line_num, len(samples), skipped)
    logger.info("Breakdown: %s", json.dumps(stats, indent=2))

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
                # Write only the messages field (what SFTTrainer needs)
                # plus metadata for traceability
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")

    # ---- Summary stats ----
    def _token_estimate(samples: list[dict]) -> int:
        """Rough char-based token estimate (÷4)."""
        total_chars = sum(
            len(m["content"])
            for s in samples
            for m in s["messages"]
        )
        return total_chars // 4

    summary = {
        "input_file": str(input_path),
        "total_raw_records": line_num,
        "converted": len(samples),
        "skipped": skipped,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_approx_tokens": _token_estimate(train_samples),
        "val_approx_tokens": _token_estimate(val_samples),
        "breakdown": stats,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
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
