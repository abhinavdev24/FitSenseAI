from __future__ import annotations

import json


def _base_params() -> dict:
    return {
        "reproducibility": {"seed": 13, "hash_seed": "seed13"},
        "phase2": {
            "synthetic": {
                "as_of_date": "2026-03-01",
                "num_users": 5,
                "profiles": {
                    "max_conditions_per_user": 2,
                    "max_medications_per_user": 2,
                    "max_allergies_per_user": 2,
                },
            }
        },
    }


def test_generate_synthetic_profiles_writes_outputs(tmp_path, module_loader):
    module = module_loader("generate_synthetic_profiles.py")
    params = _base_params()

    tables, run_dir = module.generate_synthetic_profiles(
        params=params,
        output_root=tmp_path,
        run_id="RUN_PROFILES",
    )

    assert run_dir.exists()
    assert (run_dir / "users.csv").exists()
    assert (run_dir / "user_profiles.csv").exists()

    latest = json.loads(
        (tmp_path / "synthetic_profiles" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["run_id"] == "RUN_PROFILES"
    assert latest["table_counts"]["users"] == 5

    assert len(tables["users"]) == 5
    assert "user_id" in tables["user_profiles"].columns
