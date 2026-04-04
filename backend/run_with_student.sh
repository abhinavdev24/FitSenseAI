#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-llm.txt
python3 -m uvicorn app.main:app --reload
