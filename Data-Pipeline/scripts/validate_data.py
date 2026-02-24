"""Validate distillation dataset structure and required fields."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed


REQUIRED_TOP_LEVEL = {"record_id", "instruction", "context", "response", "metadata"}
REQUIRED_CONTEXT = {"prompt_type", "slice_tags", "expected_safety_constraints", "context_summary"}
REQUIRED_SLICE_TAGS = {"age_band", "sex", "goal_type", "activity_level", "condition_flag"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_latest_distillation(raw_root: Path) -> dict[str, Any]:
    latest_path = raw_root / "distillation_dataset" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing latest distillation pointer: {latest_path}")
    return json.loads(latest_path.read_text(encoding="utf-8"))


def validate_data(params: dict[str, Any], raw_root: Path, reports_root: Path, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    latest = _load_latest_distillation(raw_root=raw_root)
    distill_dir = Path(latest["run_dir"])

    rows = _read_jsonl(distill_dir / "all_records.jsonl")
    errors: list[str] = []

    for idx, row in enumerate(rows):
        missing_top = REQUIRED_TOP_LEVEL - set(row.keys())
        if missing_top:
            errors.append(f"row[{idx}] missing top-level keys: {sorted(missing_top)}")
            continue

        context = row.get("context", {})
        if not isinstance(context, dict):
            errors.append(f"row[{idx}] context is not dict")
            continue

        missing_ctx = REQUIRED_CONTEXT - set(context.keys())
        if missing_ctx:
            errors.append(f"row[{idx}] missing context keys: {sorted(missing_ctx)}")

        slice_tags = context.get("slice_tags", {})
        if not isinstance(slice_tags, dict):
            errors.append(f"row[{idx}] slice_tags is not dict")
        else:
            missing_slice = REQUIRED_SLICE_TAGS - set(slice_tags.keys())
            if missing_slice:
                errors.append(f"row[{idx}] missing slice tags: {sorted(missing_slice)}")

        if not str(row.get("instruction", "")).strip():
            errors.append(f"row[{idx}] empty instruction")
        if not str(row.get("response", "")).strip():
            errors.append(f"row[{idx}] empty response")

    split_sizes = {}
    for split_name in ["train", "val", "test"]:
        split_file = distill_dir / f"{split_name}.jsonl"
        split_sizes[split_name] = len(_read_jsonl(split_file)) if split_file.exists() else 0

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_distillation_run_id": latest["run_id"],
        "num_rows": len(rows),
        "split_sizes": split_sizes,
        "num_errors": len(errors),
        "valid": len(errors) == 0,
        "errors": errors[:200],
    }

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = reports_root / "phase6" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate distillation dataset")
    parser.add_argument("--params", default="Data-Pipeline/params.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(int(params["reproducibility"]["seed"]), str(params["reproducibility"]["hash_seed"]))

    raw_root = Path(args.raw_root) if args.raw_root else Path(str(params["paths"]["raw_data_dir"]))
    reports_root = Path(args.reports_root) if args.reports_root else Path(str(params["paths"]["reports_dir"]))

    logger = setup_logger(
        name="fitsense.validate_data",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    report, out_path = validate_data(params=params, raw_root=raw_root, reports_root=reports_root, run_id=args.run_id)
    logger.info("Validation completed. valid=%s errors=%d output=%s", report["valid"], report["num_errors"], out_path)


if __name__ == "__main__":
    main()
