"""Reproducibility utilities for deterministic pipeline runs."""

from __future__ import annotations

import os
import random

import numpy as np


def apply_global_seed(seed: int, hash_seed: str | None = None) -> None:
    """Apply deterministic seed settings across supported libraries."""
    random.seed(seed)
    np.random.seed(seed)

    if hash_seed is not None:
        os.environ["PYTHONHASHSEED"] = hash_seed
