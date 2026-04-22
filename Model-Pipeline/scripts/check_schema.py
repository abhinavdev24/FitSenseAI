#!/usr/bin/env python3
"""
FitSenseAI — Training Data Schema Validator

Validates train.jsonl and val.jsonl files against the expected 3-turn
conversation format used for QLoRA SFT training. Referenced by the
data-pipeline-ci.yml GitHub Actions workflow.

Expected record format:
  {
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "<think>...</think>{...json...}"}
    ],
    "metadata": {...}
  }

Exit codes:
  0 — schema valid (or no files found — CI runs before training)
  1 — schema errors found
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROLES = ["system", "user", "assistant"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "Model-Pipeline" / "data" / "training" / "train.jsonl"
DEFAULT_VAL = REPO_ROOT / "Model-Pipeline" / "data" / "training" / "val.jsonl"
DEFAULT_REPORTS = REPO_ROOT / "Model-Pipeline" / "reports"


def validate_record(record: Any, line_num: int) -> list[str]:
    """Return list of error strings for a single record (empty = valid)."""
    errors: list[str] = []

    if not isinstance(record, dict):
        return [f"line {line_num}: record is not a JSON object"]

    messages = record.get("messages")
    if messages is None:
        errors.append(f"line {line_num}: missing 'messages' key")
        return errors

    if not isinstance(messages, list):
        errors.append(f"line {line_num}: 'messages' is not a list")
        return errors

    if len(messages) != 3:
        errors.append(
            f"line {line_num}: expected 3 messages, got {len(messages)}"
        )
        return errors

    for idx, (msg, expected_role) in enumerate(zip(messages, EXPECTED_ROLES)):
        if not isinstance(msg, dict):
            errors.append(f"line {line_num}: message[{idx}] is not an object")
            continue
        role = msg.get("role", "")
        if role != expected_role:
            errors.append(
                f"line {line_num}: message[{idx}] role={role!r}, "
                f"expected {expected_role!r}"
            )
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            errors.append(
                f"line {line_num}: message[{idx}] (role={role!r}) has empty content"
            )

    return errors


def validate_file(path: Path) -> dict[str, Any]:
    """Validate a single JSONL file. Returns stats dict."""
    total = 0
    errors: list[str] = []

    if not path.exists():
        print(f"  [WARN] {path} not found — skipping (expected before training)")
        return {"path": str(path), "total": 0, "valid": 0, "errors": [], "found": False}

    with open(path, encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_num}: JSON parse error — {exc}")
                continue
            errors.extend(validate_record(record, line_num))

    valid = total - len(errors)
    return {
        "path": str(path),
        "total": total,
        "valid": valid,
        "errors": errors[:50],   # cap at 50 to avoid huge reports
        "found": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FitSense training JSONL schema")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()

    print("=" * 60)
    print("FitSenseAI — Training Data Schema Validator")
    print("=" * 60)

    train_stats = validate_file(args.train_path)
    val_stats = validate_file(args.val_path)

    # Print summary
    for label, stats in [("train", train_stats), ("val", val_stats)]:
        if not stats["found"]:
            print(f"  {label}: NOT FOUND (skipped)")
        else:
            status = "✓" if not stats["errors"] else "✗"
            print(
                f"  {label}: {status}  total={stats['total']}  "
                f"valid={stats['valid']}  errors={len(stats['errors'])}"
            )
            for err in stats["errors"][:10]:
                print(f"    - {err}")
            if len(stats["errors"]) > 10:
                print(f"    ... and {len(stats['errors']) - 10} more")

    all_errors = train_stats["errors"] + val_stats["errors"]
    passed = len(all_errors) == 0
    print()
    print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")

    # Write report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.output_dir / f"schema_check_{ts}.json"
    report = {
        "timestamp": ts,
        "train_total": train_stats["total"],
        "train_valid": train_stats["valid"],
        "train_errors": train_stats["errors"],
        "val_total": val_stats["total"],
        "val_valid": val_stats["valid"],
        "val_errors": val_stats["errors"],
        "passed": passed,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report written to: {report_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
