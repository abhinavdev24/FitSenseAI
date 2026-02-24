"""Airflow DAG for FitSenseAI synthetic-data-to-distillation pipeline (Phases 1-6)."""

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
    PARAMS_PATH = REPO_ROOT / "Data-Pipeline" / "params.yaml"

    def _script_cmd(script_name: str, run_id_expr: str = "{{ ts_nodash }}") -> str:
        script_path = REPO_ROOT / "Data-Pipeline" / "scripts" / script_name
        return (
            f"cd {REPO_ROOT} && "
            f"{PYTHON_BIN} {script_path} "
            f"--params {PARAMS_PATH} "
            f"--run-id {run_id_expr}"
        )

    default_args = {
        "owner": "fitsense-mlops",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    }

    with DAG(
        dag_id="fitsense_pipeline",
        description="FitSenseAI MLOps pipeline: synthetic data -> queries -> teacher -> distillation -> QA reports",
        default_args=default_args,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        tags=["fitsense", "mlops", "synthetic", "distillation"],
    ) as dag:
        bootstrap_phase1 = BashOperator(
            task_id="bootstrap_phase1",
            bash_command=_script_cmd("bootstrap_phase1.py"),
            execution_timeout=timedelta(minutes=10),
        )

        generate_synthetic_profiles = BashOperator(
            task_id="generate_synthetic_profiles",
            bash_command=_script_cmd("generate_synthetic_profiles.py"),
            execution_timeout=timedelta(minutes=20),
        )

        generate_synthetic_workouts = BashOperator(
            task_id="generate_synthetic_workouts",
            bash_command=_script_cmd("generate_synthetic_workouts.py"),
            execution_timeout=timedelta(minutes=25),
        )

        generate_synthetic_queries = BashOperator(
            task_id="generate_synthetic_queries",
            bash_command=_script_cmd("generate_synthetic_queries.py"),
            execution_timeout=timedelta(minutes=20),
        )

        call_teacher_llm = BashOperator(
            task_id="call_teacher_llm",
            bash_command=_script_cmd("call_teacher_llm.py"),
            execution_timeout=timedelta(minutes=45),
        )

        build_distillation_dataset = BashOperator(
            task_id="build_distillation_dataset",
            bash_command=_script_cmd("build_distillation_dataset.py"),
            execution_timeout=timedelta(minutes=20),
        )

        validate_data = BashOperator(
            task_id="validate_data",
            bash_command=_script_cmd("validate_data.py"),
            execution_timeout=timedelta(minutes=10),
        )

        compute_stats = BashOperator(
            task_id="compute_stats",
            bash_command=_script_cmd("compute_stats.py"),
            execution_timeout=timedelta(minutes=10),
        )

        detect_anomalies = BashOperator(
            task_id="detect_anomalies",
            bash_command=_script_cmd("detect_anomalies.py"),
            execution_timeout=timedelta(minutes=10),
        )

        bias_slicing = BashOperator(
            task_id="bias_slicing",
            bash_command=_script_cmd("bias_slicing.py"),
            execution_timeout=timedelta(minutes=10),
        )

        (
            bootstrap_phase1
            >> generate_synthetic_profiles
            >> generate_synthetic_workouts
            >> generate_synthetic_queries
            >> call_teacher_llm
            >> build_distillation_dataset
            >> [validate_data, compute_stats, detect_anomalies, bias_slicing]
        )
