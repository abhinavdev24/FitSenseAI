"""Generate schema-aligned synthetic LLM query rows for FitSense AI.

Each row represents a prompt that will be sent to the teacher model in a
separate inference script.  This script only builds and persists the prompts;
it does NOT call any LLM.

Output fields per row
---------------------
query_id        : stable UUID derived from (prompt_type, user_id, variant_idx)
user_id         : FK → users.user_id
prompt_type     : "plan_creation" | "plan_updation"
prompt_text     : fully-rendered natural-language prompt (no system prompt)
source_run_ids  : {"synthetic_profiles": <run_id>, "synthetic_workouts": <run_id>}
created_at      : ISO-8601 timestamp (UTC)
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_uuid(kind: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"fitsense:{kind}:{value}"))


def _parse_as_of(as_of_date: str) -> date:
    return date.fromisoformat(as_of_date)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_latest_profiles(raw_root: Path) -> tuple[dict[str, pd.DataFrame], str]:
    latest_path = raw_root / "synthetic_profiles" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            "Profiles run not found. Run generate_synthetic_profiles.py first."
        )
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    run_dir = Path(payload["run_dir"])
    run_id: str = payload["run_id"]

    tables = {}
    for name in [
        "users",
        "user_profiles",
        "goals",
        "user_goals",
        "conditions",
        "user_conditions",
        "user_medical_profiles",
        "user_medications",
        "user_allergies",
    ]:
        tables[name] = pd.read_csv(run_dir / f"{name}.csv")

    return tables, run_id


def _load_latest_workouts(raw_root: Path) -> tuple[dict[str, pd.DataFrame], str]:
    latest_path = raw_root / "synthetic_workouts" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            "Workouts run not found. Run generate_synthetic_workouts.py first."
        )
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    run_dir = Path(payload["run_dir"])
    run_id: str = payload["run_id"]

    tables = {}
    for name in [
        "workout_plans",
        "plan_days",
        "plan_exercises",
        "plan_sets",
        "workouts",
        "workout_exercises",
        "workout_sets",
        "exercises",
    ]:
        tables[name] = pd.read_csv(run_dir / f"{name}.csv")

    return tables, run_id


# ---------------------------------------------------------------------------
# Per-user context builders
# ---------------------------------------------------------------------------


def _build_bio_block(
    user_id: str, profiles: dict[str, pd.DataFrame], as_of: date
) -> str:
    """Return a compact bio/medical paragraph for a single user."""
    prof_row = profiles["user_profiles"][
        profiles["user_profiles"]["user_id"] == user_id
    ]
    if prof_row.empty:
        return "No profile data available."

    r = prof_row.iloc[0]
    dob = date.fromisoformat(str(r["date_of_birth"]))
    age = (as_of - dob).days // 365

    # Goals
    user_goal_ids = (
        profiles["user_goals"][profiles["user_goals"]["user_id"] == user_id]
        .sort_values("priority")["goal_id"]
        .tolist()
    )
    goals_df = profiles["goals"]
    goal_names = goals_df[goals_df["goal_id"].isin(user_goal_ids)]["name"].tolist()

    # Conditions
    user_cond_ids = profiles["user_conditions"][
        profiles["user_conditions"]["user_id"] == user_id
    ]["condition_id"].tolist()
    cond_names = profiles["conditions"][
        profiles["conditions"]["condition_id"].isin(user_cond_ids)
    ]["name"].tolist()

    # Medical
    med_row = profiles["user_medical_profiles"][
        profiles["user_medical_profiles"]["user_id"] == user_id
    ]
    injury_text = "none"
    if not med_row.empty and bool(med_row.iloc[0]["has_injuries"]):
        injury_text = str(med_row.iloc[0]["injury_details"] or "unspecified")

    # Medications
    meds = profiles["user_medications"][
        profiles["user_medications"]["user_id"] == user_id
    ][["medication_name", "dosage", "frequency"]].to_dict("records")
    med_text = (
        ", ".join(
            f"{m['medication_name']} {m['dosage']} {m['frequency']}" for m in meds
        )
        if meds
        else "none"
    )

    # Allergies
    allergies = profiles["user_allergies"][
        profiles["user_allergies"]["user_id"] == user_id
    ]["allergen"].tolist()
    allergy_text = ", ".join(allergies) if allergies else "none"

    lines = [
        f"Age: {age}, Sex: {r['sex']}, Height: {r['height_cm']} cm",
        f"Activity level: {r['activity_level']}",
        f"Goals (priority order): {', '.join(goal_names) if goal_names else 'none'}",
        f"Medical conditions: {', '.join(cond_names) if cond_names else 'none'}",
        f"Injuries: {injury_text}",
        f"Medications: {med_text}",
        f"Allergies: {allergy_text}",
    ]
    return "\n".join(lines)


def _build_current_plan_block(user_id: str, workouts: dict[str, pd.DataFrame]) -> str:
    """Return a compact text summary of the user's active workout plan."""
    plan_row = workouts["workout_plans"][
        (workouts["workout_plans"]["user_id"] == user_id)
        & (workouts["workout_plans"]["is_active"].astype(str).str.lower() == "true")
    ]
    if plan_row.empty:
        return "No active plan found."

    plan_id = plan_row.iloc[0]["plan_id"]
    plan_name = plan_row.iloc[0]["name"]
    days = workouts["plan_days"][
        workouts["plan_days"]["plan_id"] == plan_id
    ].sort_values("day_order")

    exercises_df = workouts["exercises"]
    lines = [f"Plan: {plan_name}"]
    for _, day in days.iterrows():
        plan_day_id = day["plan_day_id"]
        lines.append(f"  Day {day['day_order']} — {day['name']}:")
        plan_exs = workouts["plan_exercises"][
            workouts["plan_exercises"]["plan_day_id"] == plan_day_id
        ].sort_values("position")
        for _, pex in plan_exs.iterrows():
            ex_name_series = exercises_df[
                exercises_df["exercise_id"] == pex["exercise_id"]
            ]["name"]
            ex_name = ex_name_series.iloc[0] if not ex_name_series.empty else "Unknown"
            sets = workouts["plan_sets"][
                workouts["plan_sets"]["plan_exercise_id"] == pex["plan_exercise_id"]
            ].sort_values("set_number")
            set_summary = "; ".join(
                f"set {s['set_number']}: {s['target_reps']} reps @ RIR {s['target_rir']}"
                for _, s in sets.iterrows()
            )
            lines.append(f"    {ex_name}: {set_summary}")
    return "\n".join(lines)


def _build_recent_workout_block(
    user_id: str,
    workouts: dict[str, pd.DataFrame],
    n_sessions: int = 3,
) -> str:
    """Return a compact text summary of the last n_sessions completed workouts."""
    user_workouts = (
        workouts["workouts"][workouts["workouts"]["user_id"] == user_id]
        .sort_values("started_at", ascending=False)
        .head(n_sessions)
    )

    if user_workouts.empty:
        return "No recorded workouts."

    exercises_df = workouts["exercises"]
    lines: list[str] = []
    for _, w in user_workouts.iterrows():
        workout_id = w["workout_id"]
        lines.append(f"Workout on {str(w['started_at'])[:10]}:")
        wx_rows = workouts["workout_exercises"][
            workouts["workout_exercises"]["workout_id"] == workout_id
        ].sort_values("position")
        for _, wx in wx_rows.iterrows():
            ex_name_series = exercises_df[
                exercises_df["exercise_id"] == wx["exercise_id"]
            ]["name"]
            ex_name = ex_name_series.iloc[0] if not ex_name_series.empty else "Unknown"
            w_sets = workouts["workout_sets"][
                workouts["workout_sets"]["workout_exercise_id"]
                == wx["workout_exercise_id"]
            ].sort_values("set_number")
            set_summary = "; ".join(
                f"set {s['set_number']}: {s['reps']} reps @ {s['weight']} kg, RIR {s['rir']}"
                for _, s in w_sets.iterrows()
            )
            lines.append(f"  {ex_name}: {set_summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt renderers
# ---------------------------------------------------------------------------

_PLAN_CREATION_VARIANTS = [
    "Generate a new workout plan for me.",
    "I want to start fresh with a new training plan.",
    "Create a weekly workout plan suited to my profile.",
    "Design a new plan optimised for my goals.",
    "Build me a workout plan from scratch.",
]

_PLAN_UPDATION_VARIANTS = [
    "Update my current plan based on how my last few sessions went.",
    "I've been feeling the workouts are too easy. Adjust the plan.",
    "My recovery has been poor lately. Dial the intensity back a bit.",
    "Swap out any exercises that stress my injuries and update my plan.",
    "I want to add more volume on push days. Update the plan accordingly.",
    "Based on my recent workouts, make my plan more progressive.",
    "I've hit a plateau. Update the plan to break it.",
]


def _render_plan_creation_prompt(bio_block: str, variant_request: str) -> str:
    return textwrap.dedent(
        f"""\
        {variant_request}

        ## My Profile
        {bio_block}

        ## Instructions
        - Do NOT include any weights in the plan.
        - Specify number of sets, target reps, and RIR (Reps In Reserve) for each set.
        - RIR scale: 0 = to failure, 1 = 1 rep left, 2 = 2 reps left, 3 = comfortable.
        - Respect all medical conditions, injuries, and medications listed above.
        - Organise the plan into named training days (e.g. PUSH_1, PULL_1, LEGS_1).
        - Return a valid JSON object and nothing else.
    """
    ).strip()


def _render_plan_updation_prompt(
    bio_block: str,
    plan_block: str,
    recent_block: str,
    variant_request: str,
) -> str:
    return textwrap.dedent(
        f"""\
        {variant_request}

        ## My Profile
        {bio_block}

        ## Current Plan
        {plan_block}

        ## Recent Workout History (last 3 sessions)
        {recent_block}

        ## Instructions
        - Do NOT include any weights in the updated plan.
        - Specify number of sets, target reps, and RIR for each set.
        - Respect all medical conditions, injuries, and medications listed above.
        - Return the full updated plan as a valid JSON object and nothing else.
    """
    ).strip()


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_synthetic_queries(
    params: dict,
    output_root: Path,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    synthetic_cfg = params["phase2"]["synthetic"]
    queries_cfg = synthetic_cfg.get("queries", {})

    seed = int(params["reproducibility"]["seed"]) + 2
    rng = np.random.default_rng(seed)
    as_of = _parse_as_of(str(synthetic_cfg["as_of_date"]))

    profiles_tables, profiles_run_id = _load_latest_profiles(output_root)
    workouts_tables, workouts_run_id = _load_latest_workouts(output_root)

    source_run_ids = {
        "synthetic_profiles": profiles_run_id,
        "synthetic_workouts": workouts_run_id,
    }

    users_df = profiles_tables["users"]
    # Optionally cap the number of users to generate queries for
    max_users: int | None = queries_cfg.get("max_users", None)
    if max_users is not None:
        users_df = users_df.head(int(max_users))

    queries_per_user_creation: int = int(
        queries_cfg.get("queries_per_user_creation", 1)
    )
    queries_per_user_updation: int = int(
        queries_cfg.get("queries_per_user_updation", 1)
    )

    rows: list[dict] = []
    created_at = datetime.now(timezone.utc).isoformat()

    for user_id in users_df["user_id"].tolist():
        bio_block = _build_bio_block(user_id, profiles_tables, as_of)
        plan_block = _build_current_plan_block(user_id, workouts_tables)
        recent_block = _build_recent_workout_block(user_id, workouts_tables)

        # ---- plan_creation queries ----
        variant_indices = rng.choice(
            len(_PLAN_CREATION_VARIANTS),
            size=queries_per_user_creation,
            replace=(
                False
                if queries_per_user_creation <= len(_PLAN_CREATION_VARIANTS)
                else True
            ),
        ).tolist()
        for vi, variant_idx in enumerate(variant_indices):
            variant_request = _PLAN_CREATION_VARIANTS[int(variant_idx)]
            prompt_text = _render_plan_creation_prompt(bio_block, variant_request)
            query_id = _stable_uuid("query", f"plan_creation:{user_id}:{vi}")
            rows.append(
                {
                    "query_id": query_id,
                    "user_id": user_id,
                    "prompt_type": "plan_creation",
                    "prompt_text": prompt_text,
                    "source_run_ids": json.dumps(source_run_ids),
                    "created_at": created_at,
                }
            )

        # ---- plan_updation queries ----
        variant_indices = rng.choice(
            len(_PLAN_UPDATION_VARIANTS),
            size=queries_per_user_updation,
            replace=(
                False
                if queries_per_user_updation <= len(_PLAN_UPDATION_VARIANTS)
                else True
            ),
        ).tolist()
        for vi, variant_idx in enumerate(variant_indices):
            variant_request = _PLAN_UPDATION_VARIANTS[int(variant_idx)]
            prompt_text = _render_plan_updation_prompt(
                bio_block, plan_block, recent_block, variant_request
            )
            query_id = _stable_uuid("query", f"plan_updation:{user_id}:{vi}")
            rows.append(
                {
                    "query_id": query_id,
                    "user_id": user_id,
                    "prompt_type": "plan_updation",
                    "prompt_text": prompt_text,
                    "source_run_ids": json.dumps(source_run_ids),
                    "created_at": created_at,
                }
            )

    queries_df = pd.DataFrame(rows)

    # ---- Persist ----
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = output_root / "synthetic_queries" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    queries_df.to_csv(run_dir / "queries.csv", index=False)

    jsonl_path = run_dir / "queries.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in queries_df.to_dict("records"):
            # Deserialise source_run_ids back to dict for JSONL
            record["source_run_ids"] = json.loads(record["source_run_ids"])
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    latest_payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "seed": seed,
        "as_of_date": str(as_of),
        "source_run_ids": source_run_ids,
        "query_counts": queries_df["prompt_type"].value_counts().to_dict(),
        "total": len(queries_df),
    }
    latest_path = output_root / "synthetic_queries" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return queries_df, run_dir


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic LLM query prompts")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument(
        "--output-root", default=None, help="Optional output root override"
    )
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(
        int(params["reproducibility"]["seed"]),
        str(params["reproducibility"]["hash_seed"]),
    )

    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(str(params["paths"]["raw_data_dir"]))
    )
    logger = setup_logger(
        name="fitsense.synthetic_queries",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    _, run_dir = generate_synthetic_queries(
        params=params, output_root=output_root, run_id=args.run_id
    )
    logger.info("Generated synthetic query data at %s", run_dir)


if __name__ == "__main__":
    main()
