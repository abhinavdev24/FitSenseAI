"""
prepare_training_data.py
------------------------
Loads distillation dataset from Phase 1 output and formats it into
<<<<<<< HEAD
Qwen3 chat template format for evaluation and inference.

Input:  Data-Pipeline/data/raw/distillation_dataset/<run_id>/{train,val,test}.jsonl
        Data-Pipeline/prompts/plan_creation.md
        Data-Pipeline/prompts/plan_updation.md
Output: Model-Pipeline/data/formatted/{train,val,test}_formatted.jsonl
=======
Qwen3 ChatML format with /no_think for structured JSON output.

Input:  Data-Pipeline/data/raw/distillation_dataset/<run_id>/{train,val,test}.jsonl
Output: Model-Pipeline/data/formatted/{run_id}/{train,val,test}_formatted.jsonl
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

DISTILLATION_BASE = Path("Data-Pipeline/data/raw/distillation_dataset")
<<<<<<< HEAD
PROMPTS_DIR       = Path("Data-Pipeline/prompts")
OUTPUT_BASE       = Path("Model-Pipeline/data/formatted")
SPLITS            = ["train", "val", "test"]
MAX_TOKENS_FLAG   = 4096
=======
OUTPUT_BASE       = Path("Model-Pipeline/data/formatted")
SPLITS            = ["train", "val", "test"]
MAX_TOKENS_FLAG   = 2048

# System prompt with explicit schema so the model knows exactly what to output.
# The schema skeleton removes ambiguity that was causing the model to invent
# keys like "workout_plan", "phases", "training_plan" etc.
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
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


<<<<<<< HEAD
# ── System prompt loader ──────────────────────────────────────────────────────

_prompt_cache: dict[str, str] = {}

def load_system_prompt(prompt_type: str) -> str:
    """
    Load system prompt from Data-Pipeline/prompts/<prompt_type>.md.
    Caches after first read. Falls back to plan_creation if type unknown.
    """
    if prompt_type in _prompt_cache:
        return _prompt_cache[prompt_type]

    prompt_file = PROMPTS_DIR / f"{prompt_type}.md"

    if not prompt_file.exists():
        log.warning(
            f"Prompt file not found: {prompt_file} — "
            f"falling back to plan_creation.md"
        )
        prompt_file = PROMPTS_DIR / "plan_creation.md"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"No prompt files found in {PROMPTS_DIR}. "
            f"Expected plan_creation.md and plan_updation.md"
        )

    content = prompt_file.read_text(encoding="utf-8").strip()
    _prompt_cache[prompt_type] = content
    log.info(f"Loaded system prompt for '{prompt_type}' from {prompt_file}")
    return content


def get_system_prompt(record: dict) -> str:
    """Select system prompt based on prompt_type in record context."""
    prompt_type = record.get("context", {}).get("prompt_type", "")
    if not prompt_type:
        log.warning("Record has no prompt_type in context — using plan_creation")
        prompt_type = "plan_creation"
    return load_system_prompt(prompt_type)


# ── Chat template ─────────────────────────────────────────────────────────────

def format_qwen3(system: str, user: str, assistant: str) -> str:
    """Qwen3 ChatML instruction format — thinking disabled via /no_think."""
    return (
        "<|im_start|>system\n"
        f"{system} /no_think\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
=======
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
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
        f"{assistant}\n"
        "<|im_end|>"
    )


def build_user_message(record: dict) -> str:
<<<<<<< HEAD
    """
    Combine instruction + context summary into the user turn.
    Falls back gracefully if context fields are missing.
    """
    instruction = record.get("instruction", "")
    context     = record.get("context", {})

    parts = [instruction]

    # append context summary if present
=======
    instruction = record.get("instruction", "")
    context     = record.get("context", {})
    parts       = [instruction]

>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    summary = context.get("context_summary") or context.get("summary", "")
    if summary:
        parts.append(f"\nContext: {summary}")

<<<<<<< HEAD
    # append safety constraints if present
=======
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    constraints = context.get("expected_safety_constraints", [])
    if constraints:
        parts.append(f"\nSafety constraints: {', '.join(constraints)}")

    return "\n".join(parts).strip()


<<<<<<< HEAD
# ── Token length estimator (no tokenizer needed) ──────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
=======
# ── Token length estimator ────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    return len(text) // 4


# ── Core processing ───────────────────────────────────────────────────────────

def process_split(input_path: Path, output_path: Path, split: str) -> dict:
<<<<<<< HEAD
    """Format one split and write to output. Returns stats."""
    records_in    = 0
    records_out   = 0
    flagged_long  = 0
    token_lengths = []
    prompt_type_counts: dict[str, int] = {}
=======
    records_in   = 0
    records_out  = 0
    flagged_long = 0
    token_lengths = []
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb

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
<<<<<<< HEAD
                log.warning(
                    f"[{split}] Skipping record with empty user/assistant: "
                    f"{record.get('record_id', '?')}"
                )
                continue

            # ── Use per-record system prompt from prompts/ directory ──
            system_prompt  = get_system_prompt(record)
            formatted_text = format_qwen3(system_prompt, user_msg, assistant)
=======
                log.warning(f"[{split}] Skipping record with empty user/assistant: {record.get('record_id', '?')}")
                continue

            formatted_text = format_qwen3(SYSTEM_PROMPT, user_msg, assistant)
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
            token_est      = estimate_tokens(formatted_text)
            token_lengths.append(token_est)

            if token_est > MAX_TOKENS_FLAG:
                flagged_long += 1

<<<<<<< HEAD
            # track prompt_type distribution
            prompt_type = record.get("context", {}).get("prompt_type", "unknown")
            prompt_type_counts[prompt_type] = (
                prompt_type_counts.get(prompt_type, 0) + 1
            )

            out_record = {
                "record_id":      record.get("record_id", ""),
                "prompt_type":    prompt_type,
=======
            out_record = {
                "record_id":      record.get("record_id", ""),
                "prompt_type":    record.get("context", {}).get("prompt_type", ""),
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
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
<<<<<<< HEAD
        "split":              split,
        "records_in":         records_in,
        "records_out":        records_out,
        "flagged_long":       flagged_long,
        "avg_tokens":         round(avg_tokens, 1),
        "max_tokens":         max_tokens,
        "prompt_type_counts": prompt_type_counts,
=======
        "split":        split,
        "records_in":   records_in,
        "records_out":  records_out,
        "flagged_long": flagged_long,
        "avg_tokens":   round(avg_tokens, 1),
        "max_tokens":   max_tokens,
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    }

    log.info(
        f"[{split}] {records_out}/{records_in} records formatted | "
<<<<<<< HEAD
        f"avg_tokens={avg_tokens:.0f} max_tokens={max_tokens} "
        f"flagged_long={flagged_long} | "
        f"prompt_types={prompt_type_counts}"
=======
        f"avg_tokens={avg_tokens:.0f} max_tokens={max_tokens} flagged_long={flagged_long}"
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    )
    return stats


# ── Runner ────────────────────────────────────────────────────────────────────

def get_latest_run_id() -> str:
<<<<<<< HEAD
    """Auto-detect the latest distillation run_id by folder mtime."""
    dirs = [d for d in DISTILLATION_BASE.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(
            f"No run directories found under {DISTILLATION_BASE}"
        )
=======
    dirs = [d for d in DISTILLATION_BASE.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under {DISTILLATION_BASE}")
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    latest = max(dirs, key=lambda d: d.stat().st_mtime)
    log.info(f"Auto-detected latest run_id: {latest.name}")
    return latest.name


def main():
<<<<<<< HEAD
    run_id           = os.environ.get("DISTILLATION_RUN_ID") or get_latest_run_id()
=======
    run_id = os.environ.get("DISTILLATION_RUN_ID") or get_latest_run_id()
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    distillation_dir = DISTILLATION_BASE / run_id
    output_dir       = OUTPUT_BASE / run_id

    log.info(f"Loading distillation dataset from: {distillation_dir}")
    log.info(f"Writing formatted output to:       {output_dir}")
<<<<<<< HEAD
    log.info(f"Loading system prompts from:       {PROMPTS_DIR}")

    # Verify prompt files exist before processing
    for prompt_type in ["plan_creation", "plan_updation"]:
        prompt_file = PROMPTS_DIR / f"{prompt_type}.md"
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Required prompt file missing: {prompt_file}\n"
                f"Make sure Data-Pipeline/prompts/ contains "
                f"plan_creation.md and plan_updation.md"
            )
        log.info(f"Found prompt file: {prompt_file}")
=======
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb

    all_stats = []

    for split in SPLITS:
        input_path  = distillation_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}_formatted.jsonl"

        if not input_path.exists():
            log.warning(f"Split file not found, skipping: {input_path}")
            continue

        stats = process_split(input_path, output_path, split)
        all_stats.append(stats)

<<<<<<< HEAD
    # Load prompt contents for manifest
    creation_prompt  = load_system_prompt("plan_creation")
    updation_prompt  = load_system_prompt("plan_updation")

    # Write manifest
    manifest = {
        "run_id":                  run_id,
        "prepared_at":             datetime.utcnow().isoformat() + "Z",
        "system_prompt_creation":  creation_prompt,
        "system_prompt_updation":  updation_prompt,
        "prompts_dir":             str(PROMPTS_DIR),
        "max_tokens_flag":         MAX_TOKENS_FLAG,
        "splits":                  all_stats,
=======
    manifest = {
        "run_id":          run_id,
        "prepared_at":     datetime.utcnow().isoformat() + "Z",
        "system_prompt":   SYSTEM_PROMPT,
        "max_tokens_flag": MAX_TOKENS_FLAG,
        "splits":          all_stats,
>>>>>>> fa3788e7322f6c3b4708c33047fa9cce653a9ffb
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Manifest written to: {manifest_path}")
    log.info("Data preparation complete.")


if __name__ == "__main__":
    main()