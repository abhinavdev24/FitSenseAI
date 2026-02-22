"""Build train/val/test distillation datasets from teacher input/output logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed


def _stable_uuid(kind: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"fitsense:{kind}:{value}"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_latest(raw_root: Path, dataset: str) -> dict[str, Any]:
    latest_path = raw_root / dataset / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing latest pointer: {latest_path}")
    return json.loads(latest_path.read_text(encoding="utf-8"))


def _hash_to_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def _stratified_split(
    df: pd.DataFrame,
    strata_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.Series:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    split = pd.Series(index=df.index, dtype="object")

    for _, idxs in df.groupby(strata_col).groups.items():
        group_df = df.loc[list(idxs)].copy()
        group_df["_u"] = group_df["record_id"].apply(_hash_to_unit_interval)
        group_df = group_df.sort_values("_u")

        n = len(group_df)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        # Keep non-empty minor splits for moderate groups.
        if n >= 3 and n_val == 0:
            n_val = 1
            n_train = max(n_train - 1, 1)
            n_test = n - n_train - n_val
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train = max(n_train - 1, 1)
            n_val = n - n_train - n_test

        train_idx = group_df.index[:n_train]
        val_idx = group_df.index[n_train : n_train + n_val]
        test_idx = group_df.index[n_train + n_val :]

        split.loc[train_idx] = "train"
        split.loc[val_idx] = "val"
        split.loc[test_idx] = "test"

    return split


def _parse_nested(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _filter_teacher_rows(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    min_chars = int(cfg["min_response_chars"])
    require_post = bool(cfg["require_post_validation"])
    reject_flags = bool(cfg["reject_on_safety_flags"])

    df = df[df["status"] == "success"].copy()
    df["response_text"] = df["response_text"].fillna("")
    df = df[df["response_text"].str.len() >= min_chars]

    df["post_validation_parsed"] = df["post_validation"].apply(lambda x: _parse_nested(x, {}))
    if require_post:
        df = df[df["post_validation_parsed"].apply(lambda x: bool(x.get("is_valid", False)))]

    if reject_flags:
        df["safety_flags_parsed"] = df["safety_flags"].apply(lambda x: _parse_nested(x, []))
        df = df[df["safety_flags_parsed"].apply(lambda x: len(x) == 0)]

    return df


def build_distillation_dataset(params: dict[str, Any], raw_root: Path, run_id: str | None = None) -> tuple[pd.DataFrame, Path]:
    cfg = dict(params["phase5"]["distillation"])
    split_cfg = dict(cfg["split"])

    teacher_meta = _load_latest(raw_root=raw_root, dataset="teacher_outputs")
    teacher_rows = _read_jsonl(Path(teacher_meta["run_dir"]) / "responses.jsonl")
    teacher_df = pd.DataFrame(teacher_rows)

    source_query_run_id = teacher_meta.get("source_query_run_id")
    if not source_query_run_id:
        raise ValueError("teacher_outputs latest.json missing source_query_run_id")

    query_jsonl = raw_root / "synthetic_queries" / source_query_run_id / "queries.jsonl"
    if not query_jsonl.exists():
        raise FileNotFoundError(f"Missing query file referenced by teacher outputs: {query_jsonl}")

    queries_df = pd.DataFrame(_read_jsonl(query_jsonl))
    filtered_teacher = _filter_teacher_rows(df=teacher_df, cfg=cfg)

    merged = filtered_teacher.merge(
        queries_df[
            [
                "query_id",
                "scenario_id",
                "user_id",
                "prompt_type",
                "slice_tags",
                "expected_safety_constraints",
                "context_summary",
                "source_run_ids",
            ]
        ],
        on=["query_id", "scenario_id", "user_id", "prompt_type"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("No rows remain after filtering/merge; check Phase 4 outputs and filter settings")

    records: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        record_id = _stable_uuid("distill_record", str(row.query_id))
        slice_tags = row.slice_tags if isinstance(row.slice_tags, dict) else _parse_nested(row.slice_tags, {})
        context_summary = (
            row.context_summary if isinstance(row.context_summary, dict) else _parse_nested(row.context_summary, {})
        )
        expected_constraints = (
            row.expected_safety_constraints
            if isinstance(row.expected_safety_constraints, list)
            else _parse_nested(row.expected_safety_constraints, [])
        )

        record = {
            "record_id": record_id,
            "query_id": row.query_id,
            "scenario_id": row.scenario_id,
            "user_id": row.user_id,
            "instruction": row.prompt_text,
            "context": {
                "prompt_type": row.prompt_type,
                "slice_tags": slice_tags,
                "expected_safety_constraints": expected_constraints,
                "context_summary": context_summary,
            },
            "response": row.response_text,
            "metadata": {
                "provider": row.provider,
                "model_name": row.model_name,
                "attempt_count": int(row.attempt_count),
                "latency_ms": int(row.latency_ms),
                "source_query_run_id": source_query_run_id,
                "created_at": _utc_now(),
            },
        }
        records.append(record)

    distill_df = pd.DataFrame(records)
    distill_df["goal_type"] = distill_df["context"].apply(
        lambda c: c.get("slice_tags", {}).get("goal_type", "unknown") if isinstance(c, dict) else "unknown"
    )
    distill_df["prompt_type"] = distill_df["context"].apply(
        lambda c: c.get("prompt_type", "unknown") if isinstance(c, dict) else "unknown"
    )
    distill_df["strata"] = distill_df["prompt_type"] + "|" + distill_df["goal_type"]

    distill_df["split"] = _stratified_split(
        df=distill_df,
        strata_col="strata",
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = raw_root / "distillation_dataset" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    train_rows = []
    val_rows = []
    test_rows = []

    for row in distill_df.itertuples(index=False):
        payload = {
            "record_id": row.record_id,
            "instruction": row.instruction,
            "context": row.context,
            "response": row.response,
            "metadata": row.metadata,
        }
        all_rows.append(payload)
        if row.split == "train":
            train_rows.append(payload)
        elif row.split == "val":
            val_rows.append(payload)
        else:
            test_rows.append(payload)

    _write_jsonl(all_rows, out_dir / "all_records.jsonl")
    _write_jsonl(train_rows, out_dir / "train.jsonl")
    _write_jsonl(val_rows, out_dir / "val.jsonl")
    _write_jsonl(test_rows, out_dir / "test.jsonl")

    summary = {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "source_teacher_run_id": teacher_meta["run_id"],
        "source_query_run_id": source_query_run_id,
        "num_all": len(all_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "filters": {
            "min_response_chars": int(cfg["min_response_chars"]),
            "require_post_validation": bool(cfg["require_post_validation"]),
            "reject_on_safety_flags": bool(cfg["reject_on_safety_flags"]),
        },
        "split": {
            "train_ratio": float(split_cfg["train_ratio"]),
            "val_ratio": float(split_cfg["val_ratio"]),
            "test_ratio": float(split_cfg["test_ratio"]),
        },
        "created_at": _utc_now(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_payload = {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "source_teacher_run_id": teacher_meta["run_id"],
        "source_query_run_id": source_query_run_id,
        "num_all": len(all_rows),
    }
    latest_path = raw_root / "distillation_dataset" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return distill_df, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build distillation dataset from teacher outputs")
    parser.add_argument("--params", default="Data-Pipeline/params.yaml", help="Path to params.yaml")
    parser.add_argument("--raw-root", default=None, help="Optional raw data root override")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(int(params["reproducibility"]["seed"]), str(params["reproducibility"]["hash_seed"]))

    raw_root = Path(args.raw_root) if args.raw_root else Path(str(params["paths"]["raw_data_dir"]))
    logger = setup_logger(
        name="fitsense.distillation",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    df, out_dir = build_distillation_dataset(params=params, raw_root=raw_root, run_id=args.run_id)
    split_counts = df["split"].value_counts().to_dict()
    logger.info("Built distillation dataset at %s with split counts: %s", out_dir, split_counts)


if __name__ == "__main__":
    main()
