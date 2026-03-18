"""Airflow DAG for FitSenseAI synthetic data generation pipeline (Phases 1-3)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.providers.standard.operators.bash import BashOperator
except ModuleNotFoundError:
    DAG = None  # type: ignore[assignment]
    BashOperator = None  # type: ignore[assignment]


if DAG is not None and BashOperator is not None:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    PYTHON_BIN = os.getenv("FITSENSE_PYTHON_BIN", "python")
    DATA_PIPELINE_DIR = REPO_ROOT / "Data-Pipeline"

    def _script_cmd(
        script_name: str,
        *,
        include_params: bool,
        include_run_id: bool,
        run_id_expr: str = "{{ ts_nodash }}",
        extra_args: str = "",
    ) -> str:
        # Scripts run directly from Data-Pipeline/; Python adds that directory to
        # sys.path automatically, so `common` is importable without PYTHONPATH.
        args: list[str] = []
        if include_params:
            args.extend(["--params", "params.yaml"])
        if include_run_id:
            args.extend(["--run-id", run_id_expr])
        if extra_args:
            args.append(extra_args.strip())

        joined_args = " ".join(args)
        return f"cd {DATA_PIPELINE_DIR} && {PYTHON_BIN} {script_name} {joined_args}".strip()

    default_args = {
        "owner": "fitsense-mlops",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    }

    with DAG(
        dag_id="fitsense_pipeline",
        description="FitSenseAI data pipeline: bootstrap -> synthetic profiles/workouts/queries",
        default_args=default_args,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        tags=["fitsense", "mlops", "synthetic"],
    ) as dag:
        bootstrap_phase1 = BashOperator(
            task_id="bootstrap_phase1",
            bash_command=_script_cmd(
                "bootstrap.py",
                include_params=False,
                include_run_id=False,
            ),
            execution_timeout=timedelta(minutes=10),
        )

        generate_synthetic_profiles = BashOperator(
            task_id="generate_synthetic_profiles",
            bash_command=_script_cmd(
                "generate_synthetic_profiles.py",
                include_params=True,
                include_run_id=True,
            ),
            execution_timeout=timedelta(minutes=20),
        )

        generate_synthetic_workouts = BashOperator(
            task_id="generate_synthetic_workouts",
            bash_command=_script_cmd(
                "generate_synthetic_workouts.py",
                include_params=True,
                include_run_id=True,
            ),
            execution_timeout=timedelta(minutes=25),
        )

        generate_synthetic_queries = BashOperator(
            task_id="generate_synthetic_queries",
            bash_command=_script_cmd(
                "generate_synthetic_queries.py",
                include_params=True,
                include_run_id=True,
            ),
            execution_timeout=timedelta(minutes=20),
        )

        bootstrap_phase1.set_downstream(generate_synthetic_profiles)
        generate_synthetic_profiles.set_downstream(generate_synthetic_workouts)
        generate_synthetic_workouts.set_downstream(generate_synthetic_queries)
