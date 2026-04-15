from __future__ import annotations

import os
from urllib.parse import unquote
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_ENGINE = (os.getenv("DATABASE_ENGINE") or "").strip().lower()
DATABASE_USER = (os.getenv("DATABASE_USER") or "").strip()
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD") or ""
DATABASE_NAME = (os.getenv("DATABASE_NAME") or "").strip()
DATABASE_HOST = (os.getenv("DATABASE_HOST") or "").strip()
DATABASE_PORT = (os.getenv("DATABASE_PORT") or "").strip()
DATABASE_PATH = (os.getenv("DATABASE_PATH") or "").strip()

def _build_database_url() -> URL:
    if DATABASE_ENGINE == "sqlite":
        if not DATABASE_PATH:
            raise RuntimeError("DATABASE_PATH is required when DATABASE_ENGINE=sqlite")
        return URL.create("sqlite", database=DATABASE_PATH)

    if DATABASE_ENGINE and DATABASE_ENGINE not in {"mysql", "mysql+pymysql"}:
        raise RuntimeError(
            f"Unsupported DATABASE_ENGINE value: {DATABASE_ENGINE}. Use sqlite or mysql."
        )

    missing_values = [
        name
        for name, value in {
            "DATABASE_USER": DATABASE_USER,
            "DATABASE_PASSWORD": DATABASE_PASSWORD,
            "DATABASE_NAME": DATABASE_NAME,
            "DATABASE_HOST": DATABASE_HOST,
            "DATABASE_PORT": DATABASE_PORT,
        }.items()
        if not value
    ]
    if missing_values:
        raise RuntimeError(
            "Missing database environment variables: " + ", ".join(missing_values)
        )

    try:
        port = int(DATABASE_PORT)
    except ValueError as exc:
        raise RuntimeError(f"DATABASE_PORT must be an integer, got: {DATABASE_PORT}") from exc

    return URL.create(
        "mysql+pymysql",
        username=DATABASE_USER,
        password=unquote(DATABASE_PASSWORD),
        host=DATABASE_HOST,
        port=port,
        database=DATABASE_NAME,
    )

engine_kwargs = {"future": True}
database_url = _build_database_url()

if database_url.drivername.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
elif database_url.drivername.startswith("mysql"):
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["connect_args"] = {"ssl": {"check_hostname": False}}
else:
    raise RuntimeError(
        f"Unsupported DATABASE_ENGINE scheme: {database_url.drivername}. "
        "Use sqlite or mysql+pymysql."
    )

engine = create_engine(database_url, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
