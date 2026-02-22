"""Configuration loader for the FitSenseAI pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_PARAMS_PATH = Path("Data-Pipeline/params.yaml")


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load pipeline parameters from YAML."""
    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")

    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    if not isinstance(params, dict):
        raise ValueError("params.yaml must parse to a top-level dictionary")

    return params
