import os
import sys

from dotenv import load_dotenv

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.local")

DATABASE_ENGINE = (os.getenv("DATABASE_ENGINE") or "").strip().lower()
DATABASE_PATH = (os.getenv("DATABASE_PATH") or "").strip()

if DATABASE_ENGINE != "sqlite":
    print(
        "reset_db.py only supports explicit SQLite databases. "
        "Set DATABASE_ENGINE=sqlite and DATABASE_PATH to run it.",
        file=sys.stderr,
    )
    sys.exit(1)

if not DATABASE_PATH:
    print("DATABASE_PATH is required when DATABASE_ENGINE=sqlite.", file=sys.stderr)
    sys.exit(1)

DB_PATH = Path(DATABASE_PATH)
if DB_PATH.exists():
    DB_PATH.unlink()
    print(f"Deleted {DB_PATH}")
else:
    print(f"No database found at {DB_PATH}")
