"""Build action-specific system prompts from base + action schema files.

Usage:
    from prompt_builder import build_system_prompt

    prompt = build_system_prompt("plan_creation")
    # Returns: base rules + plan_creation schema only
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Map action names to their schema filenames
ACTION_SCHEMA_FILES: dict[str, str] = {
    "plan_creation": "action_plan_creation.md",
    "plan_modification": "action_plan_modification.md",
    "safety_adjustment": "action_safety_adjustment.md",
    "log_workout": "action_workout_logging.md",
    "log_weight": "action_log_weight.md",
    "log_calories": "action_log_calories.md",
    "log_sleep": "action_log_sleep.md",
    "progress_adaptation": "action_progress_adaptation.md",
    "progress_comment": "action_progress_comment.md",
    "coaching_qa": "action_coaching_qa.md",
}

# Actions that were previously grouped under "metric_logging"
METRIC_ACTION_MAP: dict[str, str] = {
    "weight": "log_weight",
    "sleep": "log_sleep",
    "calories": "log_calories",
}


def build_system_prompt(
    action: str,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Compose a system prompt from the base prompt + the single relevant action schema.

    Args:
        action: The action key (e.g. "plan_creation", "log_weight").
        prompts_dir: Directory containing prompt files. Defaults to ./prompts/

    Returns:
        Combined system prompt string.

    Raises:
        ValueError: If action is not recognized.
        FileNotFoundError: If prompt files are missing.
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent.parent / "prompts"

    if action not in ACTION_SCHEMA_FILES:
        raise ValueError(
            f"Unknown action: {action!r}. "
            f"Valid actions: {sorted(ACTION_SCHEMA_FILES.keys())}"
        )

    base_path = prompts_dir / "coach_base.md"
    action_path = prompts_dir / ACTION_SCHEMA_FILES[action]

    if not base_path.exists():
        raise FileNotFoundError(f"Missing base prompt: {base_path}")
    if not action_path.exists():
        raise FileNotFoundError(f"Missing action schema: {action_path}")

    base = base_path.read_text(encoding="utf-8").strip()
    schema = action_path.read_text(encoding="utf-8").strip()

    return f"{base}\n\n{schema}"


def resolve_metric_action(metric_type: str) -> str:
    """Convert a legacy metric type string to the new granular action name.

    Args:
        metric_type: One of "weight", "sleep", "calories".

    Returns:
        The resolved action key (e.g. "log_weight").
    """
    resolved = METRIC_ACTION_MAP.get(metric_type)
    if resolved is None:
        raise ValueError(
            f"Unknown metric type: {metric_type!r}. "
            f"Valid types: {sorted(METRIC_ACTION_MAP.keys())}"
        )
    return resolved
