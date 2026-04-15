"""Pytest configuration for backend tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.local")

TEST_DB_PATH = Path(tempfile.gettempdir()) / "fitsense_test.db"
TEST_DB_PATH.unlink(missing_ok=True)
os.environ["DATABASE_ENGINE"] = "sqlite"
os.environ["DATABASE_PATH"] = str(TEST_DB_PATH)
