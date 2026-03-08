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
    min_chars = int(cfg.get("min_response_chars", 0))
    require_post = bool(cfg.get("require_post_validation", False))
    reject_flags = bool(cfg.get("reject_on_safety_flags", False))

    # Basic success filter
    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()

    if "response_text" in df.columns:
        df["response_text"] = df["response_text"].fillna("")
        df = df[df["response_text"].str.len() >= min_chars]

    if "post_validation" in df.columns and require_post:
        df["post_validation_parsed"] = df["post_validation"].apply(
            lambda x: _parse_nested(x, {})
        )
        df = df[
            df["post_validation_parsed"].apply(lambda x: bool(x.get("is_valid", False)))
        ]

    if "safety_flags" in df.columns and reject_flags:
        df["safety_flags_parsed"] = df["safety_flags"].apply(
            lambda x: _parse_nested(x, [])
        )
        df = df[df["safety_flags_parsed"].apply(lambda x: len(x) == 0)]

    return df


def build_distillation_dataset(
    params: dict[str, Any], raw_root: Path, run_id: str | None = None
) -> tuple[pd.DataFrame, Path]:
    cfg = dict(params["phase5"]["distillation"])
    split_cfg = dict(cfg["split"])

    teacher_meta = _load_latest(raw_root=raw_root, dataset="teacher_outputs")
    filename = "responses.jsonl"

    teacher_rows = _read_jsonl(Path(teacher_meta["run_dir"]) / filename)
    teacher_df = pd.DataFrame(teacher_rows)

    # Note: synthetic_queries metadata might be in teacher_meta or nested source_query_run_id
    source_query_run_id = teacher_meta.get("source_query_run_id")

    queries_df = None
    if source_query_run_id:
        query_jsonl = (
            raw_root / "synthetic_queries" / source_query_run_id / "queries.jsonl"
        )
        if query_jsonl.exists():
            queries_df = pd.DataFrame(_read_jsonl(query_jsonl))

    filtered_teacher = _filter_teacher_rows(df=teacher_df, cfg=cfg)

    # Merge with queries if available to get extra context tags
    if queries_df is not None and "query_id" in filtered_teacher.columns:
        merged = filtered_teacher.merge(
            queries_df[
                [
                    "query_id",
                    "scenario_id",
                    "user_id",
                    "slice_tags",
                    "expected_safety_constraints",
                    "context_summary",
                ]
            ],
            on=["query_id"],
            how="left",
        )
    else:
        merged = filtered_teacher

    if merged.empty:
        raise ValueError("No rows remain after filtering; check teacher outputs")

    records: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        record_id = _stable_uuid(
            "distill_record", str(getattr(row, "query_id", row.record_id))
        )

        # Handle both old flattened and new conversation formats
        if hasattr(row, "messages"):
            messages = row.messages
            instruction = messages[1]["content"] if len(messages) > 1 else ""
            response = (
                messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
            )
        else:
            instruction = getattr(row, "prompt_text", getattr(row, "instruction", ""))
            response = getattr(row, "response_text", getattr(row, "response", ""))
            messages = [
                {"role": "system", "content": "You are FitSenseAI. Respond ONLY with a valid JSON object."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]

        slice_tags = _parse_nested(getattr(row, "slice_tags", {}), {})
        context_summary = _parse_nested(getattr(row, "context_summary", {}), {})
        expected_constraints = _parse_nested(
            getattr(row, "expected_safety_constraints", []), []
        )

        record = {
            "record_id": record_id,
            "prompt_type": getattr(row, "prompt_type", "unknown"),
            "messages": messages,
            "response_json": _parse_nested(getattr(row, "response_json", None), None),
            "instruction": instruction,
            "response": response,
            "context": {
                "slice_tags": slice_tags,
                "expected_safety_constraints": expected_constraints,
                "context_summary": context_summary,
            },
            "metadata": {
                "model_name": getattr(row, "model_name", "teacher"),
                "created_at": _utc_now(),
            },
        }
        records.append(record)

    distill_df = pd.DataFrame(records)

    # Stratified split based on prompt_type and goal (if available)
    distill_df["goal_type"] = distill_df["context"].apply(
        lambda c: c.get("slice_tags", {}).get("goal_type", "unknown")
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

    splits = ["train", "val", "test"]
    all_rows = records
    _write_jsonl(all_rows, out_dir / "all_records.jsonl")

    for s in splits:
        split_records = [
            r for r, split_val in zip(records, distill_df["split"]) if split_val == s
        ]
        _write_jsonl(split_records, out_dir / f"{s}.jsonl")

    # Write summary
    summary = {
        "run_id": run_id,
        "num_all": len(all_rows),
        "splits": distill_df["split"].value_counts().to_dict(),
        "created_at": _utc_now(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    latest_path = raw_root / "distillation_dataset" / "latest.json"
    latest_path.write_text(
        json.dumps({"run_id": run_id, "run_dir": str(out_dir)}, indent=2)
    )

    return distill_df, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build distillation dataset from teacher outputs"
    )
    parser.add_argument(
        "--params", default="params.yaml", help="Path to params.yaml"
    )
    parser.add_argument(
        "--raw-root", default=None, help="Optional raw data root override"
    )
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(
        int(params["reproducibility"]["seed"]),
        str(params["reproducibility"]["hash_seed"]),
    )

    raw_root = (
        Path(args.raw_root)
        if args.raw_root
        else Path(str(params["paths"]["raw_data_dir"]))
    )
    logger = setup_logger(
        name="fitsense.distillation",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    df, out_dir = build_distillation_dataset(
        params=params, raw_root=raw_root, run_id=args.run_id
    )
    split_counts = df["split"].value_counts().to_dict()
    logger.info(
        "Built distillation dataset at %s with split counts: %s", out_dir, split_counts
    )


if __name__ == "__main__":
    main()
