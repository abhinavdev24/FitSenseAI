"""Detect anomalies in distillation dataset and emit alert-style report."""

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


def detect_anomalies(params: dict[str, Any], raw_root: Path, reports_root: Path, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    cfg = params["phase6"]["anomaly_detection"]

    latest = _load_latest_distillation(raw_root=raw_root)
    distill_dir = Path(latest["run_dir"])
    rows = _read_jsonl(distill_dir / "all_records.jsonl")
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No distillation rows found")

    df["response_len"] = df["response"].astype(str).str.len()

    duplicates = int(df["record_id"].duplicated().sum())
    missing_response = int((df["response"].astype(str).str.strip() == "").sum())

    min_chars = int(cfg["min_response_chars"])
    max_chars = int(cfg["max_response_chars"])
    short_responses = int((df["response_len"] < min_chars).sum())
    long_responses = int((df["response_len"] > max_chars).sum())

    split_sizes = {s: len(_read_jsonl(distill_dir / f"{s}.jsonl")) for s in ["train", "val", "test"]}
    total = max(sum(split_sizes.values()), 1)
    observed_ratios = {k: v / total for k, v in split_sizes.items()}

    summary_path = distill_dir / "summary.json"
    expected = {"train": 0.8, "val": 0.1, "test": 0.1}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "train": float(summary.get("split", {}).get("train_ratio", 0.8)),
            "val": float(summary.get("split", {}).get("val_ratio", 0.1)),
            "test": float(summary.get("split", {}).get("test_ratio", 0.1)),
        }

    tolerance = float(cfg["split_ratio_tolerance"])
    split_deviation = {
        k: abs(observed_ratios.get(k, 0.0) - expected.get(k, 0.0))
        for k in ["train", "val", "test"]
    }

    anomalies = {
        "duplicate_records": duplicates > int(cfg["duplicate_record_threshold"]),
        "missing_responses": missing_response > int(cfg["missing_response_threshold"]),
        "short_responses": short_responses > 0,
        "long_responses": long_responses > 0,
        "split_imbalance": any(v > tolerance for v in split_deviation.values()),
    }

    severity = "none"
    if any(anomalies.values()):
        severity = "warning"
    if anomalies["duplicate_records"] or anomalies["missing_responses"]:
        severity = "critical"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_distillation_run_id": latest["run_id"],
        "counts": {
            "num_rows": int(len(df)),
            "duplicate_records": duplicates,
            "missing_responses": missing_response,
            "short_responses": short_responses,
            "long_responses": long_responses,
        },
        "split": {
            "sizes": split_sizes,
            "observed_ratios": observed_ratios,
            "expected_ratios": expected,
            "deviation": split_deviation,
            "tolerance": tolerance,
        },
        "anomalies": anomalies,
        "severity": severity,
        "alerts": [k for k, v in anomalies.items() if v],
    }

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = reports_root / "phase6" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anomaly_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect anomalies in distillation dataset")
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
        name="fitsense.detect_anomalies",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    report, out_path = detect_anomalies(params=params, raw_root=raw_root, reports_root=reports_root, run_id=args.run_id)
    logger.info("Anomaly detection complete. severity=%s alerts=%s output=%s", report["severity"], report["alerts"], out_path)


if __name__ == "__main__":
    main()
