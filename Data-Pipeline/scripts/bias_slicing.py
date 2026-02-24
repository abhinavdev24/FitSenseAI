"""Run bias slicing analysis on distillation dataset."""

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


def _summarize_slice(df: pd.DataFrame, group_col: str, min_group_size: int) -> dict[str, Any]:
    grp = (
        df.groupby(group_col)
        .agg(
            n=("record_id", "count"),
            mean_response_len=("response_len", "mean"),
        )
        .reset_index()
    )

    grp = grp[grp["n"] >= min_group_size]
    if grp.empty:
        return {"group_col": group_col, "groups": [], "max_gap": 0.0}

    max_gap = float(grp["mean_response_len"].max() - grp["mean_response_len"].min())
    groups = [
        {
            "group": str(r[group_col]),
            "n": int(r["n"]),
            "mean_response_len": float(r["mean_response_len"]),
        }
        for _, r in grp.iterrows()
    ]
    return {"group_col": group_col, "groups": groups, "max_gap": max_gap}


def bias_slicing(params: dict[str, Any], raw_root: Path, reports_root: Path, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    cfg = params["phase6"]["bias_slicing"]
    min_group_size = int(cfg["min_group_size"])
    max_gap_allowed = float(cfg["max_mean_response_len_gap"])

    latest = _load_latest_distillation(raw_root=raw_root)
    rows = _read_jsonl(Path(latest["run_dir"]) / "all_records.jsonl")
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No distillation rows found")

    df["response_len"] = df["response"].astype(str).str.len()
    df["age_band"] = df["context"].apply(lambda c: c.get("slice_tags", {}).get("age_band", "unknown"))
    df["sex"] = df["context"].apply(lambda c: c.get("slice_tags", {}).get("sex", "unknown"))
    df["goal_type"] = df["context"].apply(lambda c: c.get("slice_tags", {}).get("goal_type", "unknown"))
    df["activity_level"] = df["context"].apply(lambda c: c.get("slice_tags", {}).get("activity_level", "unknown"))
    df["condition_flag"] = df["context"].apply(lambda c: c.get("slice_tags", {}).get("condition_flag", "unknown"))

    slice_reports = []
    for col in ["age_band", "sex", "goal_type", "activity_level", "condition_flag"]:
        slice_reports.append(_summarize_slice(df=df, group_col=col, min_group_size=min_group_size))

    flagged = [
        {
            "group_col": r["group_col"],
            "max_gap": r["max_gap"],
            "threshold": max_gap_allowed,
        }
        for r in slice_reports
        if r["max_gap"] > max_gap_allowed
    ]

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_distillation_run_id": latest["run_id"],
        "num_rows": int(len(df)),
        "min_group_size": min_group_size,
        "max_mean_response_len_gap_threshold": max_gap_allowed,
        "slice_reports": slice_reports,
        "flagged_slices": flagged,
        "bias_alert": len(flagged) > 0,
    }

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = reports_root / "phase6" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bias slicing analysis")
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
        name="fitsense.bias_slicing",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    report, out_path = bias_slicing(params=params, raw_root=raw_root, reports_root=reports_root, run_id=args.run_id)
    logger.info("Bias slicing complete. bias_alert=%s output=%s", report["bias_alert"], out_path)


if __name__ == "__main__":
    main()
