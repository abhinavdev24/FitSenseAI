"""Generate schema-aligned synthetic user/profile/goal/condition tables."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed

GOALS = [
    ("fat_loss", "Reduce body fat while preserving lean mass."),
    ("muscle_gain", "Increase muscle mass progressively."),
    ("strength", "Improve compound lift performance."),
    ("general_fitness", "Improve overall health and activity consistency."),
]

CONDITIONS = [
    ("none", "No known chronic medical condition."),
    ("hypertension", "Elevated blood pressure requiring monitoring."),
    ("type2_diabetes", "Type 2 diabetes managed with lifestyle/medication."),
    ("asthma", "Respiratory condition with intermittent symptoms."),
    ("lower_back_pain", "Recurring lower back pain requiring exercise modifications."),
    ("knee_pain", "Chronic knee discomfort requiring movement constraints."),
]

ACTIVITY_LEVELS = ["sedentary", "light", "moderate", "active", "very_active"]
SEX_OPTIONS = ["male", "female", "non_binary"]
SEVERITIES = ["mild", "moderate", "high"]

FIRST_NAMES = [
    "Alex",
    "Jordan",
    "Taylor",
    "Riley",
    "Casey",
    "Sam",
    "Avery",
    "Drew",
    "Morgan",
    "Jamie",
]
LAST_NAMES = [
    "Shah",
    "Kim",
    "Patel",
    "Singh",
    "Brown",
    "Chen",
    "Garcia",
    "Miller",
    "Johnson",
    "Nguyen",
]


def _stable_uuid(kind: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"fitsense:{kind}:{value}"))


def _parse_as_of(as_of_date: str) -> date:
    return date.fromisoformat(as_of_date)


def _create_users(num_users: int, as_of: date, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    users_rows: list[dict[str, object]] = []
    profiles_rows: list[dict[str, object]] = []

    for idx in range(1, num_users + 1):
        user_key = f"user_{idx:05d}"
        user_id = _stable_uuid("user", user_key)

        first_name = FIRST_NAMES[int(rng.integers(0, len(FIRST_NAMES)))]
        last_name = LAST_NAMES[int(rng.integers(0, len(LAST_NAMES)))]
        created_days_ago = int(rng.integers(10, 720))
        created_at = datetime.combine(as_of - timedelta(days=created_days_ago), datetime.min.time()).replace(tzinfo=timezone.utc)

        users_rows.append(
            {
                "user_id": user_id,
                "name": f"{first_name} {last_name}",
                "email": f"{first_name.lower()}.{last_name.lower()}.{idx}@fitsense.synthetic",
                "created_at": created_at.isoformat(),
            }
        )

        age_years = int(rng.integers(18, 66))
        dob = as_of - timedelta(days=age_years * 365 + int(rng.integers(0, 365)))
        sex = SEX_OPTIONS[int(rng.integers(0, len(SEX_OPTIONS)))]
        height_cm = round(float(rng.normal(171.0, 10.0)), 1)
        height_cm = min(max(height_cm, 145.0), 205.0)
        activity_level = ACTIVITY_LEVELS[int(rng.integers(0, len(ACTIVITY_LEVELS)))]

        profiles_rows.append(
            {
                "user_id": user_id,
                "date_of_birth": dob.isoformat(),
                "sex": sex,
                "height_cm": height_cm,
                "activity_level": activity_level,
                "updated_at": created_at.isoformat(),
            }
        )

    return pd.DataFrame(users_rows), pd.DataFrame(profiles_rows)


def _create_goals_and_links(users_df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    goals_df = pd.DataFrame(
        [{"goal_id": _stable_uuid("goal", name), "name": name, "description": desc} for name, desc in GOALS]
    )

    links: list[dict[str, object]] = []
    goal_ids = goals_df["goal_id"].tolist()

    for user_id in users_df["user_id"].tolist():
        goal_count = int(rng.integers(1, 3))
        selected = rng.choice(goal_ids, size=goal_count, replace=False)
        for priority, goal_id in enumerate(selected, start=1):
            links.append({"user_id": user_id, "goal_id": str(goal_id), "priority": priority})

    return goals_df, pd.DataFrame(links)


def _create_conditions_and_links(
    users_df: pd.DataFrame,
    rng: np.random.Generator,
    max_conditions_per_user: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conditions_df = pd.DataFrame(
        [
            {"condition_id": _stable_uuid("condition", name), "name": name, "description": desc}
            for name, desc in CONDITIONS
        ]
    )

    user_conditions_rows: list[dict[str, object]] = []
    medical_rows: list[dict[str, object]] = []

    condition_ids = conditions_df.loc[conditions_df["name"] != "none", "condition_id"].tolist()

    for user_id in users_df["user_id"].tolist():
        has_injuries = bool(rng.random() < 0.25)
        medical_rows.append(
            {
                "medical_profile_id": _stable_uuid("medical_profile", user_id),
                "user_id": user_id,
                "has_injuries": has_injuries,
                "injury_details": "knee strain" if has_injuries else None,
                "surgeries_history": None,
                "family_history": None,
                "notes": "synthetic_record",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        condition_count = int(rng.integers(0, max_conditions_per_user + 1))
        if condition_count == 0:
            continue

        selected = rng.choice(condition_ids, size=condition_count, replace=False)
        for condition_id in selected:
            user_conditions_rows.append(
                {
                    "user_id": user_id,
                    "condition_id": str(condition_id),
                    "severity": SEVERITIES[int(rng.integers(0, len(SEVERITIES)))],
                    "notes": "synthetic_condition",
                }
            )

    return conditions_df, pd.DataFrame(user_conditions_rows), pd.DataFrame(medical_rows)


def _create_medications(users_df: pd.DataFrame, rng: np.random.Generator, max_medications_per_user: int) -> pd.DataFrame:
    medications = ["Metformin", "Lisinopril", "Albuterol", "Ibuprofen"]
    frequencies = ["daily", "twice_daily", "as_needed"]
    rows: list[dict[str, object]] = []

    for user_id in users_df["user_id"].tolist():
        med_count = int(rng.integers(0, max_medications_per_user + 1))
        for idx in range(med_count):
            med_name = medications[int(rng.integers(0, len(medications)))]
            rows.append(
                {
                    "medication_id": _stable_uuid("medication", f"{user_id}:{idx}:{med_name}"),
                    "user_id": user_id,
                    "medication_name": med_name,
                    "dosage": f"{int(rng.integers(5, 501))} mg",
                    "frequency": frequencies[int(rng.integers(0, len(frequencies)))],
                    "start_date": "2025-01-01",
                    "end_date": None,
                    "notes": "synthetic_medication",
                }
            )

    return pd.DataFrame(rows)


def _create_allergies(users_df: pd.DataFrame, rng: np.random.Generator, max_allergies_per_user: int) -> pd.DataFrame:
    allergens = ["peanuts", "shellfish", "lactose", "pollen", "dust"]
    reactions = ["rash", "swelling", "digestive discomfort", "sneezing"]
    rows: list[dict[str, object]] = []

    for user_id in users_df["user_id"].tolist():
        allergy_count = int(rng.integers(0, max_allergies_per_user + 1))
        for idx in range(allergy_count):
            allergen = allergens[int(rng.integers(0, len(allergens)))]
            rows.append(
                {
                    "allergy_id": _stable_uuid("allergy", f"{user_id}:{idx}:{allergen}"),
                    "user_id": user_id,
                    "allergen": allergen,
                    "reaction": reactions[int(rng.integers(0, len(reactions)))],
                    "severity": SEVERITIES[int(rng.integers(0, len(SEVERITIES)))],
                    "notes": "synthetic_allergy",
                }
            )

    return pd.DataFrame(rows)


def _create_targets(users_df: pd.DataFrame, profiles_df: pd.DataFrame, as_of: date, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = users_df[["user_id"]].merge(profiles_df[["user_id", "activity_level"]], on="user_id", how="left")
    calories_rows: list[dict[str, object]] = []
    sleep_rows: list[dict[str, object]] = []

    level_adjustment = {
        "sedentary": -250,
        "light": -100,
        "moderate": 0,
        "active": 200,
        "very_active": 350,
    }

    for row in merged.itertuples(index=False):
        user_id = row.user_id
        adjustment = level_adjustment.get(row.activity_level, 0)
        maintenance = int(2200 + adjustment + rng.integers(-150, 151))
        maintenance = int(min(max(maintenance, 1400), 3800))

        calories_rows.append(
            {
                "calorie_target_id": _stable_uuid("calorie_target", user_id),
                "user_id": user_id,
                "maintenance_calories": maintenance,
                "method": "synthetic_estimate",
                "effective_from": as_of.isoformat(),
                "effective_to": None,
                "notes": "synthetic_target",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        sleep_hours = round(float(rng.normal(7.6, 0.6)), 2)
        sleep_hours = min(max(sleep_hours, 6.0), 9.5)
        sleep_rows.append(
            {
                "sleep_target_id": _stable_uuid("sleep_target", user_id),
                "user_id": user_id,
                "target_sleep_hours": sleep_hours,
                "effective_from": as_of.isoformat(),
                "effective_to": None,
                "notes": "synthetic_target",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return pd.DataFrame(calories_rows), pd.DataFrame(sleep_rows)


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, df in tables.items():
        df.to_csv(output_dir / f"{table_name}.csv", index=False)


def generate_synthetic_profiles(params: dict, output_root: Path, run_id: str | None = None) -> tuple[dict[str, pd.DataFrame], Path]:
    synthetic_cfg = params["phase2"]["synthetic"]
    profiles_cfg = synthetic_cfg["profiles"]

    seed = int(params["reproducibility"]["seed"])
    rng = np.random.default_rng(seed)
    as_of = _parse_as_of(str(synthetic_cfg["as_of_date"]))
    num_users = int(synthetic_cfg["num_users"])

    users_df, user_profiles_df = _create_users(num_users=num_users, as_of=as_of, rng=rng)
    goals_df, user_goals_df = _create_goals_and_links(users_df=users_df, rng=rng)
    conditions_df, user_conditions_df, medical_profiles_df = _create_conditions_and_links(
        users_df=users_df,
        rng=rng,
        max_conditions_per_user=int(profiles_cfg["max_conditions_per_user"]),
    )
    medications_df = _create_medications(
        users_df=users_df,
        rng=rng,
        max_medications_per_user=int(profiles_cfg["max_medications_per_user"]),
    )
    allergies_df = _create_allergies(
        users_df=users_df,
        rng=rng,
        max_allergies_per_user=int(profiles_cfg["max_allergies_per_user"]),
    )
    calorie_targets_df, sleep_targets_df = _create_targets(users_df=users_df, profiles_df=user_profiles_df, as_of=as_of, rng=rng)

    tables: dict[str, pd.DataFrame] = {
        "users": users_df,
        "user_profiles": user_profiles_df,
        "goals": goals_df,
        "user_goals": user_goals_df,
        "conditions": conditions_df,
        "user_conditions": user_conditions_df,
        "user_medical_profiles": medical_profiles_df,
        "user_medications": medications_df,
        "user_allergies": allergies_df,
        "calorie_targets": calorie_targets_df,
        "sleep_targets": sleep_targets_df,
    }

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_dir = output_root / "synthetic_profiles" / run_id
    _write_tables(tables=tables, output_dir=run_dir)

    latest_path = output_root / "synthetic_profiles" / "latest.json"
    latest_payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "seed": seed,
        "as_of_date": str(as_of),
        "table_counts": {name: int(len(df)) for name, df in tables.items()},
    }
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return tables, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic profiles dataset")
    parser.add_argument("--params", default="Data-Pipeline/params.yaml", help="Path to params.yaml")
    parser.add_argument("--output-root", default=None, help="Optional output root override")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(int(params["reproducibility"]["seed"]), str(params["reproducibility"]["hash_seed"]))

    output_root = Path(args.output_root) if args.output_root else Path(str(params["paths"]["raw_data_dir"]))
    logger = setup_logger(
        name="fitsense.synthetic_profiles",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    _, run_dir = generate_synthetic_profiles(params=params, output_root=output_root, run_id=args.run_id)
    logger.info("Generated synthetic profiles data at %s", run_dir)


if __name__ == "__main__":
    main()
