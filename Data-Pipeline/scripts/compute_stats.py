"""Compute descriptive statistics for distillation dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed


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


def compute_stats(params: dict[str, Any], raw_root: Path, reports_root: Path, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    latest = _load_latest_distillation(raw_root=raw_root)
    distill_dir = Path(latest["run_dir"])
    rows = _read_jsonl(distill_dir / "all_records.jsonl")

    if not rows:
        raise ValueError("Distillation dataset is empty")

    df = pd.DataFrame(rows)
    df["prompt_type"] = df["context"].apply(lambda c: c.get("prompt_type", "unknown"))
    df["goal_type"] = df["context"].apply(lambda c: c.get("slice_tags", {}).get("goal_type", "unknown"))
    df["activity_level"] = df["context"].apply(lambda c: c.get("slice_tags", {}).get("activity_level", "unknown"))
    df["response_len"] = df["response"].astype(str).str.len()

    split_sizes = {}
    for split in ["train", "val", "test"]:
        split_rows = _read_jsonl(distill_dir / f"{split}.jsonl")
        split_sizes[split] = len(split_rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_distillation_run_id": latest["run_id"],
        "num_rows": int(len(df)),
        "split_sizes": split_sizes,
        "prompt_type_counts": df["prompt_type"].value_counts().to_dict(),
        "goal_type_counts": df["goal_type"].value_counts().to_dict(),
        "activity_level_counts": df["activity_level"].value_counts().to_dict(),
        "response_length": {
            "min": int(df["response_len"].min()),
            "p50": float(df["response_len"].median()),
            "p95": float(df["response_len"].quantile(0.95)),
            "max": int(df["response_len"].max()),
            "mean": float(df["response_len"].mean()),
        },
    }

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = reports_root / "phase6" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stats_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute distillation dataset stats")
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
        name="fitsense.compute_stats",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    report, out_path = compute_stats(params=params, raw_root=raw_root, reports_root=reports_root, run_id=args.run_id)
    logger.info("Stats computed for %d rows. output=%s", report["num_rows"], out_path)


if __name__ == "__main__":
    main()
