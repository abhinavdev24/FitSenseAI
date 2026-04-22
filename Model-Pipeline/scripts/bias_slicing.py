#!/usr/bin/env python3
"""
FitSenseAI — Training Data Bias Slicer

CPU-only analysis of val.jsonl. Extracts demographic attributes from the
user message via regex, counts records per slice, and flags slices that
are under-represented (< 5%) or over-represented (> 40%).

Referenced by the data-pipeline-ci.yml GitHub Actions workflow.

Exit code: always 0 (informational — not a hard gate).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAL = REPO_ROOT / "Model-Pipeline" / "data" / "training" / "val.jsonl"
DEFAULT_REPORTS = REPO_ROOT / "Model-Pipeline" / "reports"

# Thresholds for flagging
MIN_FRACTION = 0.05   # slices below this are under-represented
MAX_FRACTION = 0.40   # slices above this are over-represented


# ---------------------------------------------------------------------------
# Attribute extractors
# ---------------------------------------------------------------------------

def extract_age_group(text: str) -> str:
    m = re.search(r"[Aa]ge[:\s]+(\d+)", text)
    if not m:
        return "unknown"
    age = int(m.group(1))
    if age <= 25:
        return "18-25"
    if age <= 35:
        return "26-35"
    if age <= 50:
        return "36-50"
    return "50+"


def extract_sex(text: str) -> str:
    m = re.search(r"[Ss]ex[:\s]+(male|female|M|F|other)\b", text, re.IGNORECASE)
    if not m:
        return "unknown"
    raw = m.group(1).lower()
    if raw in ("m", "male"):
        return "male"
    if raw in ("f", "female"):
        return "female"
    return "other"


def extract_goal_type(text: str) -> str:
    """Extract the first goal listed after 'Goals'."""
    m = re.search(r"[Gg]oals?\s*(?:\(.*?\))?\s*[:\-]\s*([^\n,]+)", text)
    if not m:
        return "unknown"
    raw = m.group(1).strip().lower()
    # Normalise common goal names
    if any(k in raw for k in ("muscle", "strength", "hypertrophy")):
        return "muscle_gain"
    if any(k in raw for k in ("weight loss", "fat loss", "cut")):
        return "weight_loss"
    if any(k in raw for k in ("endurance", "cardio", "cycling", "running")):
        return "endurance"
    if any(k in raw for k in ("general", "fitness", "health")):
        return "general_fitness"
    return raw.split("_")[0][:20]  # truncate to avoid huge keys


def extract_activity_level(text: str) -> str:
    m = re.search(
        r"[Aa]ctivity\s+level[:\s]+(sedentary|lightly_active|moderately_active|very_active)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return "unknown"
    return m.group(1).lower()


EXTRACTORS: dict[str, Any] = {
    "age_group": extract_age_group,
    "sex": extract_sex,
    "goal_type": extract_goal_type,
    "activity_level": extract_activity_level,
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_user_content(record: dict[str, Any]) -> str:
    for msg in record.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def slice_dataset(path: Path) -> tuple[int, dict[str, dict[str, int]]]:
    """Return (total_records, {dimension: {slice_value: count}})."""
    counts: dict[str, dict[str, int]] = {dim: defaultdict(int) for dim in EXTRACTORS}
    total = 0

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            user_text = get_user_content(record)
            for dim, fn in EXTRACTORS.items():
                val = fn(user_text)
                counts[dim][val] += 1

    return total, {dim: dict(v) for dim, v in counts.items()}


def find_flagged(
    total: int, slices: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for dim, values in slices.items():
        for val, count in values.items():
            fraction = count / total if total else 0.0
            if fraction < MIN_FRACTION:
                flagged.append(
                    {
                        "dimension": dim,
                        "value": val,
                        "count": count,
                        "fraction": round(fraction, 4),
                        "reason": f"under-represented (< {MIN_FRACTION:.0%})",
                    }
                )
            elif fraction > MAX_FRACTION:
                flagged.append(
                    {
                        "dimension": dim,
                        "value": val,
                        "count": count,
                        "fraction": round(fraction, 4),
                        "reason": f"over-represented (> {MAX_FRACTION:.0%})",
                    }
                )
    return flagged


def main() -> int:
    parser = argparse.ArgumentParser(description="FitSense training data bias slicer")
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()

    print("=" * 60)
    print("FitSenseAI — Training Data Bias Slicer")
    print("=" * 60)

    if not args.val_path.exists():
        print(f"  [WARN] {args.val_path} not found — skipping (expected before training)")
        # Write empty report and exit cleanly
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = args.output_dir / f"bias_report_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"timestamp": ts, "total_records": 0, "slices": {}, "flagged": []},
                fh,
                indent=2,
            )
        print(f"Report written to: {report_path}")
        return 0

    total, slices = slice_dataset(args.val_path)
    flagged = find_flagged(total, slices)

    # Print per-dimension summary
    for dim, values in slices.items():
        print(f"\n  {dim}:")
        for val, count in sorted(values.items(), key=lambda x: -x[1]):
            frac = count / total if total else 0.0
            marker = "  "
            if frac < MIN_FRACTION:
                marker = "⚠ "
            elif frac > MAX_FRACTION:
                marker = "⚠ "
            print(f"  {marker}  {val:<25} {count:>4}  ({frac:.1%})")

    print()
    if flagged:
        print(f"  Flagged slices: {len(flagged)}")
        for item in flagged:
            print(f"  - {item['dimension']}/{item['value']}: {item['reason']}")
    else:
        print("  No imbalanced slices detected.")

    # Write report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.output_dir / f"bias_report_{ts}.json"
    report = {
        "timestamp": ts,
        "total_records": total,
        "slices": {
            dim: {val: cnt for val, cnt in values.items()}
            for dim, values in slices.items()
        },
        "flagged": flagged,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to: {report_path}")

    return 0  # always informational — never a hard gate


if __name__ == "__main__":
    sys.exit(main())
