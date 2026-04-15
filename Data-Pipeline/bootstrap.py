"""Phase 1 bootstrap script.

Validates config/logging/reproducibility plumbing and writes a sanity artifact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed


def main() -> None:
    params = load_params()

    seed = int(params["reproducibility"]["seed"])
    hash_seed = str(params["reproducibility"]["hash_seed"])
    apply_global_seed(seed=seed, hash_seed=hash_seed)

    logger = setup_logger(
        name="fitsense.phase1",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    report_path = Path(str(params["paths"]["reports_dir"])) / "phase1_bootstrap.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "project": params["project"]["name"],
        "seed": seed,
        "hash_seed": hash_seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "phase": "phase1_scaffold",
    }

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Phase 1 bootstrap completed. Wrote %s", report_path)


if __name__ == "__main__":
    main()
