"""
prepare_training_data.py
------------------------
Loads distillation dataset from Phase 1 output and formats it into
Qwen3 ChatML format with /no_think for structured JSON output.

Input:  Data-Pipeline/data/raw/distillation_dataset/<run_id>/{train,val,test}.jsonl
Output: Model-Pipeline/data/formatted/{run_id}/{train,val,test}_formatted.jsonl
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

DISTILLATION_BASE = Path("Data-Pipeline/data/raw/distillation_dataset")
OUTPUT_BASE       = Path("Model-Pipeline/data/formatted")
SPLITS            = ["train", "val", "test"]
MAX_TOKENS_FLAG   = 2048

SYSTEM_PROMPT = (
    "You are FitSense AI, an expert fitness coach and periodization specialist. "
    "You provide personalised, safety-aware workout plans and coaching guidance. "
    "When generating workout plans, return ONLY valid JSON with this exact structure:\n"
    '{"plan_name": "...", "days": [{"name": "...", "day_order": 1, "notes": null, '
    '"exercises": [{"exercise_name": "...", "position": 1, "notes": null, '
    '"sets": [{"set_number": 1, "target_reps": 10, "target_rir": 2, "rest_seconds": 60}]}]}]}\n'
    "No other top-level keys. No markdown fences. No prose before or after the JSON. "
    "Always respect the user's medical conditions, injuries, and constraints."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── Chat template ─────────────────────────────────────────────────────────────

def format_qwen3(system: str, user: str, assistant: str) -> str:
    """
    Qwen3 ChatML format with /no_think appended to the user turn and an
    empty <think></think> block injected before the assistant response.
    This trains the model to skip the reasoning phase and output JSON directly.
    """
    return (
        "<|im_start|>system\n"
        f"{system}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n</think>\n"
        f"{assistant}\n"
        "<|im_end|>"
    )


def build_user_message(record: dict) -> str:
    instruction = record.get("instruction", "")
    context     = record.get("context", {})
    parts       = [instruction]

    summary = context.get("context_summary") or context.get("summary", "")
    if summary:
        parts.append(f"\nContext: {summary}")

    constraints = context.get("expected_safety_constraints", [])
    if constraints:
        parts.append(f"\nSafety constraints: {', '.join(constraints)}")

    return "\n".join(parts).strip()


# ── Token length estimator ────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return len(text) // 4


# ── Core processing ───────────────────────────────────────────────────────────

def process_split(input_path: Path, output_path: Path, split: str) -> dict:
    records_in    = 0
    records_out   = 0
    flagged_long  = 0
    token_lengths = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records_in += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"[{split}] Skipping malformed record: {e}")
                continue

            user_msg  = build_user_message(record)
            assistant = record.get("response", "")

            if not user_msg or not assistant:
                log.warning(f"[{split}] Skipping record with empty user/assistant: {record.get('record_id', '?')}")
                continue

            formatted_text = format_qwen3(SYSTEM_PROMPT, user_msg, assistant)
            token_est      = estimate_tokens(formatted_text)
            token_lengths.append(token_est)

            if token_est > MAX_TOKENS_FLAG:
                flagged_long += 1

            out_record = {
                "record_id":      record.get("record_id", ""),
                "prompt_type":    record.get("context", {}).get("prompt_type", ""),
                "goal_type":      record.get("context", {}).get("slice_tags", {}).get("goal_type", ""),
                "condition_flag": record.get("context", {}).get("slice_tags", {}).get("condition_flag", ""),
                "activity_level": record.get("context", {}).get("slice_tags", {}).get("activity_level", ""),
                "age_band":       record.get("context", {}).get("slice_tags", {}).get("age_band", ""),
                "sex":            record.get("context", {}).get("slice_tags", {}).get("sex", ""),
                "formatted_text": formatted_text,
                "user_message":   user_msg,
                "assistant":      assistant,
                "token_estimate": token_est,
                "flagged_long":   token_est > MAX_TOKENS_FLAG,
            }

            fout.write(json.dumps(out_record) + "\n")
            records_out += 1

    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0

    stats = {
        "split":        split,
        "records_in":   records_in,
        "records_out":  records_out,
        "flagged_long": flagged_long,
        "avg_tokens":   round(avg_tokens, 1),
        "max_tokens":   max_tokens,
    }

    log.info(
        f"[{split}] {records_out}/{records_in} records formatted | "
        f"avg_tokens={avg_tokens:.0f} max_tokens={max_tokens} flagged_long={flagged_long}"
    )
    return stats


# ── Runner ────────────────────────────────────────────────────────────────────

def get_latest_run_id() -> str:
    dirs = [d for d in DISTILLATION_BASE.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under {DISTILLATION_BASE}")
    latest = max(dirs, key=lambda d: d.stat().st_mtime)
    log.info(f"Auto-detected latest run_id: {latest.name}")
    return latest.name


def main():
    run_id           = os.environ.get("DISTILLATION_RUN_ID") or get_latest_run_id()
    distillation_dir = DISTILLATION_BASE / run_id
    output_dir       = OUTPUT_BASE / run_id

    log.info(f"Loading distillation dataset from: {distillation_dir}")
    log.info(f"Writing formatted output to:       {output_dir}")

    all_stats = []

    for split in SPLITS:
        input_path  = distillation_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}_formatted.jsonl"

        if not input_path.exists():
            log.warning(f"Split file not found, skipping: {input_path}")
            continue

        stats = process_split(input_path, output_path, split)
        all_stats.append(stats)

    manifest = {
        "run_id":          run_id,
        "prepared_at":     datetime.utcnow().isoformat() + "Z",
        "system_prompt":   SYSTEM_PROMPT,
        "max_tokens_flag": MAX_TOKENS_FLAG,
        "splits":          all_stats,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Manifest written to: {manifest_path}")
    log.info("Data preparation complete.")


if __name__ == "__main__":
    main()