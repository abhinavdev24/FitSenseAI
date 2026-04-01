from __future__ import annotations

import re
from pathlib import Path


def test_dags_package_docstring(module_loader):
    module = module_loader("dags/__init__.py")
    assert module.__doc__ is not None
    assert "FitSenseAI" in module.__doc__


def test_fitsense_pipeline_module_loads_without_airflow(module_loader):
    module = module_loader("dags/fitsense_pipeline.py")

    # In environments without Airflow the module should still import cleanly.
    if module.DAG is None:
        assert getattr(module, "BashOperator") is None
        return

    assert hasattr(module, "dag")
    task = module.dag.get_task("generate_synthetic_profiles")
    cmd = task.bash_command
    assert "--params params.yaml" in cmd
    assert "--run-id {{ ts_nodash }}" in cmd


def test_fitsense_pipeline_references_existing_scripts_static():
    pipeline_root = Path(__file__).resolve().parents[1]
    dag_file = pipeline_root / "dags" / "fitsense_pipeline.py"
    dag_text = dag_file.read_text(encoding="utf-8")

    referenced = set(re.findall(r"\b([a-zA-Z0-9_]+\.py)\b", dag_text))
    referenced.discard("params.yaml")

    expected_scripts = {
        "bootstrap.py",
        "generate_synthetic_profiles.py",
        "generate_synthetic_workouts.py",
        "generate_synthetic_queries.py",
    }
    assert expected_scripts.issubset(referenced)

    for script in expected_scripts:
        assert (pipeline_root / script).exists(), f"Missing script: {script}"
