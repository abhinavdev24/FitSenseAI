"""Generate synthetic teacher-ready prompts from synthetic profile/workout data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed


def _stable_uuid(kind: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"fitsense:{kind}:{value}"))


def _load_latest_run(raw_root: Path, dataset_name: str) -> tuple[dict, Path]:
    latest_path = raw_root / dataset_name / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing latest run pointer: {latest_path}")

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    run_dir = Path(payload["run_dir"])
    return payload, run_dir


def _age_band(age: int) -> str:
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def _build_user_context(raw_root: Path) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    _, profile_dir = _load_latest_run(raw_root=raw_root, dataset_name="synthetic_profiles")
    _, workout_dir = _load_latest_run(raw_root=raw_root, dataset_name="synthetic_workouts")

    users = pd.read_csv(profile_dir / "users.csv")
    user_profiles = pd.read_csv(profile_dir / "user_profiles.csv")
    goals = pd.read_csv(profile_dir / "goals.csv")
    user_goals = pd.read_csv(profile_dir / "user_goals.csv")
    conditions = pd.read_csv(profile_dir / "conditions.csv")
    user_conditions = pd.read_csv(profile_dir / "user_conditions.csv")

    workouts = pd.read_csv(workout_dir / "workouts.csv")
    workout_exercises = pd.read_csv(workout_dir / "workout_exercises.csv")
    workout_sets = pd.read_csv(workout_dir / "workout_sets.csv")

    goal_name_map = dict(zip(goals["goal_id"], goals["name"]))
    cond_name_map = dict(zip(conditions["condition_id"], conditions["name"]))

    primary_goals = (
        user_goals.sort_values(["user_id", "priority"]).drop_duplicates("user_id", keep="first")[
            ["user_id", "goal_id"]
        ]
    )
    primary_goals["primary_goal"] = primary_goals["goal_id"].map(goal_name_map)

    user_conditions["condition_name"] = user_conditions["condition_id"].map(cond_name_map)
    cond_agg = (
        user_conditions.groupby("user_id")["condition_name"]
        .apply(lambda s: sorted([x for x in s.dropna().unique().tolist()]))
        .reset_index(name="conditions")
    )

    workout_counts = workouts.groupby("user_id").size().reset_index(name="workout_count")

    ws = workout_sets.merge(
        workout_exercises[["workout_exercise_id", "workout_id"]],
        on="workout_exercise_id",
        how="inner",
    ).merge(workouts[["workout_id", "user_id"]], on="workout_id", how="inner")

    perf = (
        ws.groupby("user_id")
        .agg(
            avg_reps=("reps", "mean"),
            avg_weight=("weight", "mean"),
            avg_rir=("rir", "mean"),
        )
        .reset_index()
    )

    context = (
        users[["user_id", "name"]]
        .merge(user_profiles[["user_id", "date_of_birth", "sex", "activity_level"]], on="user_id", how="left")
        .merge(primary_goals[["user_id", "primary_goal"]], on="user_id", how="left")
        .merge(cond_agg, on="user_id", how="left")
        .merge(workout_counts, on="user_id", how="left")
        .merge(perf, on="user_id", how="left")
    )

    context["conditions"] = context["conditions"].apply(lambda x: x if isinstance(x, list) else [])
    context["workout_count"] = context["workout_count"].fillna(0).astype(int)
    context["avg_reps"] = context["avg_reps"].fillna(0.0)
    context["avg_weight"] = context["avg_weight"].fillna(0.0)
    context["avg_rir"] = context["avg_rir"].fillna(0.0)

    return context, goal_name_map, cond_name_map


def _compute_age(dob_str: str) -> int:
    dob = datetime.fromisoformat(dob_str).date()
    today = datetime.now(timezone.utc).date()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return max(age, 18)


def _scenario_prompt(prompt_type: str, context_row: pd.Series, rng: np.random.Generator) -> tuple[str, list[str]]:
    goal = context_row["primary_goal"] or "general_fitness"
    conditions = context_row["conditions"]
    cond_text = ", ".join(conditions) if conditions else "none"

    safety_constraints = ["avoid unsafe load spikes", "prioritize progressive overload"]
    if conditions:
        safety_constraints.append("respect medical constraints and low-impact alternatives")

    if prompt_type == "plan_creation":
        prompt = (
            f"Create a 7-day workout plan for a user with goal '{goal}', activity level '{context_row['activity_level']}', "
            f"conditions: {cond_text}. Include sets, reps, RIR, and rest guidance."
        )
    elif prompt_type == "plan_modification":
        prompt = (
            f"Modify an existing plan for user goal '{goal}'. Recent averages: reps={context_row['avg_reps']:.1f}, "
            f"weight={context_row['avg_weight']:.1f} kg, RIR={context_row['avg_rir']:.1f}. "
            "Adjust intensity and exercise order for next week."
        )
    elif prompt_type == "safety_adjustment":
        prompt = (
            f"Given conditions {cond_text}, produce safer substitutions and loading limits for a workout plan. "
            "Highlight contraindicated movements and alternatives."
        )
    elif prompt_type == "progress_adaptation":
        trend = rng.choice(["plateau", "improving", "fatigue_signals"])
        prompt = (
            f"User trend is '{trend}' with goal '{goal}'. Propose progression or deload adjustments for the next 2 weeks, "
            "and explain why."
        )
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    return prompt, safety_constraints


def _write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def generate_synthetic_queries(params: dict, raw_root: Path, run_id: str | None = None) -> tuple[pd.DataFrame, Path]:
    cfg = params["phase3"]["synthetic_queries"]
    prompt_types: list[str] = list(cfg["prompt_types"])
    prompts_per_type = int(cfg["prompts_per_type"])

    if prompts_per_type < 1:
        raise ValueError("phase3.synthetic_queries.prompts_per_type must be >= 1")

    seed = int(params["reproducibility"]["seed"]) + 2
    rng = np.random.default_rng(seed)

    profile_meta, _ = _load_latest_run(raw_root=raw_root, dataset_name="synthetic_profiles")
    workout_meta, _ = _load_latest_run(raw_root=raw_root, dataset_name="synthetic_workouts")
    context_df, _, _ = _build_user_context(raw_root=raw_root)

    now_iso = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    for _, row in context_df.iterrows():
        user_id = row["user_id"]
        age = _compute_age(str(row["date_of_birth"]))
        age_band = _age_band(age)
        conditions = row["conditions"] if isinstance(row["conditions"], list) else []
        condition_flag = "has_condition" if conditions else "none"

        for prompt_type in prompt_types:
            for variant_idx in range(prompts_per_type):
                scenario_id = _stable_uuid("scenario", f"{user_id}:{prompt_type}:{variant_idx}")
                query_id = _stable_uuid("query", f"{scenario_id}")
                prompt_text, safety_constraints = _scenario_prompt(
                    prompt_type=prompt_type,
                    context_row=row,
                    rng=rng,
                )

                record = {
                    "query_id": query_id,
                    "scenario_id": scenario_id,
                    "user_id": user_id,
                    "prompt_type": prompt_type,
                    "prompt_variant": variant_idx,
                    "prompt_text": prompt_text,
                    "slice_tags": {
                        "age_band": age_band,
                        "sex": row["sex"],
                        "goal_type": row["primary_goal"] or "general_fitness",
                        "activity_level": row["activity_level"],
                        "condition_flag": condition_flag,
                    },
                    "expected_safety_constraints": safety_constraints,
                    "context_summary": {
                        "workout_count": int(row["workout_count"]),
                        "avg_reps": round(float(row["avg_reps"]), 2),
                        "avg_weight": round(float(row["avg_weight"]), 2),
                        "avg_rir": round(float(row["avg_rir"]), 2),
                        "conditions": conditions,
                    },
                    "source_run_ids": {
                        "synthetic_profiles": profile_meta["run_id"],
                        "synthetic_workouts": workout_meta["run_id"],
                    },
                    "created_at": now_iso,
                }
                records.append(record)

    queries_df = pd.DataFrame(records)

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = raw_root / "synthetic_queries" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "queries.jsonl"
    csv_path = run_dir / "queries.csv"
    _write_jsonl(records=records, output_path=jsonl_path)
    queries_df.to_csv(csv_path, index=False)

    latest_payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "seed": seed,
        "num_queries": int(len(queries_df)),
        "prompt_types": prompt_types,
        "prompts_per_type": prompts_per_type,
        "source_run_ids": {
            "synthetic_profiles": profile_meta["run_id"],
            "synthetic_workouts": workout_meta["run_id"],
        },
    }

    latest_path = raw_root / "synthetic_queries" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return queries_df, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic teacher prompts from synthetic data")
    parser.add_argument("--params", default="Data-Pipeline/params.yaml", help="Path to params.yaml")
    parser.add_argument("--raw-root", default=None, help="Optional raw data root override")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(int(params["reproducibility"]["seed"]), str(params["reproducibility"]["hash_seed"]))

    raw_root = Path(args.raw_root) if args.raw_root else Path(str(params["paths"]["raw_data_dir"]))
    logger = setup_logger(
        name="fitsense.synthetic_queries",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    queries_df, run_dir = generate_synthetic_queries(params=params, raw_root=raw_root, run_id=args.run_id)
    logger.info("Generated %d synthetic queries at %s", len(queries_df), run_dir)


if __name__ == "__main__":
    main()
