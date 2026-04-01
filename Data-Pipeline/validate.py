"""
validate.py
===========
Validates that a response record's response_json matches the expected
workout plan structure produced by the teacher LLM.

Expected structure
------------------
{
  "plan_name": str,
  "days": [
    {
      "name": str,
      "day_order": int,
      "notes": str | null,
      "exercises": [
        {
          "exercise_name": str,
          "position": int,
          "notes": str | null,
          "sets": [
            {
              "set_number": int,
              "target_reps": int,
              "target_rir": int,
              "rest_seconds": int
            }
          ]
        }
      ]
    }
  ]
}

Rules
-----
- response_json must be a dict (not None, not a list, not a string)
- Must have "plan_name" (non-empty string) and "days" (non-empty list)
- Each day must have: name (str), day_order (int), exercises (list)
  - Rest/recovery days may have an empty exercises list
- Each exercise must have: exercise_name (str), position (int), sets (non-empty list)
- Each set must have: set_number (int), target_reps (int), target_rir (int), rest_seconds (int)
  - null values are coerced to safe defaults (e.g. target_rir: null → 0)
- target_weight must NOT be present in any set (hard rule from system prompt)

Usage as a module
-----------------
    from validate import validate_response_json, ValidationResult

    result = validate_response_json(record["response_json"])
    if not result.ok:
        print(result.reason)

Usage as a CLI tool
-------------------
    python validate.py responses.jsonl
    python validate.py responses.jsonl --fix          # rewrites file, moves bad to failed
    python validate.py responses.jsonl --fix \\
        --failed-path failed_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


_OK = ValidationResult(ok=True)


def _fail(reason: str) -> ValidationResult:
    return ValidationResult(ok=False, reason=reason)


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------


def validate_response_json(response_json: Any) -> ValidationResult:
    """
    Validate that response_json matches the expected workout plan structure.
    Returns ValidationResult(ok=True) on success or ValidationResult(ok=False, reason=...)
    on the first structural violation found.
    """
    if response_json is None:
        return _fail("response_json is None")

    if not isinstance(response_json, dict):
        return _fail(
            f"response_json is not a dict (got {type(response_json).__name__})"
        )

    # Top-level fields
    plan_name = response_json.get("plan_name")
    if not isinstance(plan_name, str) or not plan_name.strip():
        return _fail("missing or empty 'plan_name'")

    days = response_json.get("days")
    if not isinstance(days, list) or len(days) == 0:
        return _fail("'days' must be a non-empty list")

    for day_idx, day in enumerate(days):
        if not isinstance(day, dict):
            return _fail(f"days[{day_idx}] is not a dict")

        if not isinstance(day.get("name"), str) or not day["name"].strip():
            return _fail(f"days[{day_idx}] missing or empty 'name'")

        if not isinstance(day.get("day_order"), int):
            return _fail(f"days[{day_idx}] 'day_order' must be an int")

        exercises = day.get("exercises")
        if not isinstance(exercises, list):
            return _fail(f"days[{day_idx}] 'exercises' must be a list")

        # Rest / recovery days may legitimately have an empty exercise list.
        _REST_KEYWORDS = ("rest", "recovery", "off", "deload")
        day_name_lower = (day.get("name") or "").lower()
        day_notes_lower = (day.get("notes") or "").lower()
        _is_rest_day = any(
            kw in day_name_lower or kw in day_notes_lower for kw in _REST_KEYWORDS
        )
        if len(exercises) == 0:
            if _is_rest_day:
                continue  # valid rest day — skip exercise validation
            return _fail(f"days[{day_idx}] 'exercises' is empty on a non-rest day")

        for ex_idx, ex in enumerate(exercises):
            if not isinstance(ex, dict):
                return _fail(f"days[{day_idx}].exercises[{ex_idx}] is not a dict")

            if (
                not isinstance(ex.get("exercise_name"), str)
                or not ex["exercise_name"].strip()
            ):
                return _fail(
                    f"days[{day_idx}].exercises[{ex_idx}] missing or empty 'exercise_name'"
                )

            if not isinstance(ex.get("position"), int):
                return _fail(
                    f"days[{day_idx}].exercises[{ex_idx}] 'position' must be an int"
                )

            sets = ex.get("sets")
            if not isinstance(sets, list) or len(sets) == 0:
                return _fail(
                    f"days[{day_idx}].exercises[{ex_idx}] 'sets' must be a non-empty list"
                )

            for set_idx, s in enumerate(sets):
                if not isinstance(s, dict):
                    return _fail(
                        f"days[{day_idx}].exercises[{ex_idx}].sets[{set_idx}] is not a dict"
                    )

                loc = f"days[{day_idx}].exercises[{ex_idx}].sets[{set_idx}]"

                # Coerce null / missing values to safe defaults.
                # The LLM sometimes emits null for stretches/cardio where
                # a field like target_rir has no meaningful value.
                _SET_DEFAULTS = {
                    "set_number": set_idx + 1,
                    "target_reps": 1,
                    "target_rir": 0,
                    "rest_seconds": 0,
                }
                for field, default in _SET_DEFAULTS.items():
                    val = s.get(field)
                    if val is None:
                        s[field] = default
                    elif not isinstance(val, int):
                        return _fail(f"{loc} '{field}' must be an int (got {val!r})")

                if "target_weight" in s:
                    return _fail(f"{loc} contains forbidden field 'target_weight'")

    return _OK


# ---------------------------------------------------------------------------
# Record-level validator (operates on a full response record)
# ---------------------------------------------------------------------------


def validate_record(record: dict) -> ValidationResult:
    """
    Validate a full response record.
    A record is invalid if:
      - status is not "success"
      - response_json fails structural validation
    """
    status = record.get("status", "")
    if status != "success":
        return _fail(f"status is '{status}' (not 'success')")

    return validate_response_json(record.get("response_json"))


# ---------------------------------------------------------------------------
# File-level validator / fixer
# ---------------------------------------------------------------------------


def validate_jsonl_file(
    responses_path: Path,
    failed_path: Path,
    fix: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Read responses_path, validate every record.

    Returns:
        valid_records   — records that passed validation
        invalid_records — records that failed validation
        unreadable      — lines that couldn't be parsed as JSON

    If fix=True:
        - Rewrites responses_path with only valid_records
        - Appends invalid_records to failed_path (deduplicating by response_id)
    """
    if not responses_path.exists():
        return [], [], []

    valid_records: list[dict] = []
    invalid_records: list[dict] = []
    unreadable: list[dict] = []

    with responses_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                unreadable.append({"line": lineno, "error": str(e), "raw": line[:200]})
                continue

            result = validate_record(rec)
            if result.ok:
                valid_records.append(rec)
            else:
                rec["_validation_error"] = result.reason
                invalid_records.append(rec)

    if fix:
        # Rewrite responses.jsonl with only valid records
        with responses_path.open("w", encoding="utf-8") as fh:
            for rec in valid_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Append invalid to failed_responses.jsonl, deduplicating by response_id
        existing_failed_ids: set[str] = set()
        if failed_path.exists():
            with failed_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            existing_failed_ids.add(r.get("response_id", ""))
                        except json.JSONDecodeError:
                            pass

        with failed_path.open("a", encoding="utf-8") as fh:
            for rec in invalid_records:
                rid = rec.get("response_id", "")
                if rid not in existing_failed_ids:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    existing_failed_ids.add(rid)

    return valid_records, invalid_records, unreadable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate responses.jsonl against the workout plan schema"
    )
    parser.add_argument("responses_path", help="Path to responses.jsonl")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite responses.jsonl keeping only valid records; "
        "move invalid to failed_responses.jsonl",
    )
    parser.add_argument(
        "--failed-path",
        default=None,
        help="Path to failed_responses.jsonl (default: sibling of responses_path)",
    )
    args = parser.parse_args()

    responses_path = Path(args.responses_path)
    if not responses_path.exists():
        print(f"ERROR: file not found: {responses_path}", file=sys.stderr)
        sys.exit(1)

    failed_path = (
        Path(args.failed_path)
        if args.failed_path
        else responses_path.parent / "failed_responses.jsonl"
    )

    valid, invalid, unreadable = validate_jsonl_file(
        responses_path=responses_path,
        failed_path=failed_path,
        fix=args.fix,
    )

    total = len(valid) + len(invalid) + len(unreadable)
    print(f"\nValidation report: {responses_path}")
    print(f"  total lines   : {total}")
    print(f"  valid         : {len(valid)}")
    print(f"  invalid       : {len(invalid)}")
    print(f"  unreadable    : {len(unreadable)}")

    if invalid:
        print(f"\nInvalid records:")
        for rec in invalid:
            print(
                f"  query_id={rec.get('query_id', '?')!s:40s}  "
                f"reason={rec.get('_validation_error', '?')}"
            )

    if unreadable:
        print(f"\nUnreadable lines:")
        for u in unreadable:
            print(f"  line {u['line']}: {u['error']}")

    if args.fix:
        print(f"\nFix applied:")
        print(f"  responses.jsonl    rewritten with {len(valid)} valid records")
        print(f"  failed_responses.jsonl appended with {len(invalid)} invalid records")

    sys.exit(0 if not invalid and not unreadable else 1)


if __name__ == "__main__":
    main()
