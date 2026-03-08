"""Logging helpers for the FitSenseAI pipeline."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, level: str, log_dir: str, file_name: str, fmt: str) -> logging.Logger:
    """Create a logger configured for both stdout and file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / file_name)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
