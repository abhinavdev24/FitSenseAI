from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "fitsense.db"
if DB_PATH.exists():
    DB_PATH.unlink()
    print(f"Deleted {DB_PATH}")
else:
    print(f"No database found at {DB_PATH}")
