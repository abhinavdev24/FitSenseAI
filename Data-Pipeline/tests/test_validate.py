from __future__ import annotations

import json


def _valid_response_json() -> dict:
    return {
        "plan_name": "Starter Plan",
        "days": [
            {
                "name": "PUSH_1",
                "day_order": 1,
                "notes": None,
                "exercises": [
                    {
                        "exercise_name": "Push Up",
                        "position": 1,
                        "notes": None,
                        "sets": [
                            {
                                "set_number": 1,
                                "target_reps": 10,
                                "target_rir": 2,
                                "rest_seconds": 90,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_validate_response_json_success(module_loader):
    module = module_loader("validate.py")
    result = module.validate_response_json(_valid_response_json())
    assert result.ok is True


def test_validate_response_json_rejects_target_weight(module_loader):
    module = module_loader("validate.py")
    payload = _valid_response_json()
    payload["days"][0]["exercises"][0]["sets"][0]["target_weight"] = 20.0

    result = module.validate_response_json(payload)
    assert result.ok is False
    assert "target_weight" in result.reason


def test_validate_jsonl_file_fix_rewrites_and_moves_invalid(tmp_path, module_loader):
    module = module_loader("validate.py")

    responses_path = tmp_path / "responses.jsonl"
    failed_path = tmp_path / "failed_responses.jsonl"

    valid_record = {
        "response_id": "r1",
        "query_id": "q1",
        "status": "success",
        "response_json": _valid_response_json(),
    }
    invalid_record = {
        "response_id": "r2",
        "query_id": "q2",
        "status": "failed",
        "response_json": {},
    }

    responses_path.write_text(
        "\n".join(
            [
                json.dumps(valid_record),
                json.dumps(invalid_record),
                "{not-json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    valid, invalid, unreadable = module.validate_jsonl_file(
        responses_path=responses_path,
        failed_path=failed_path,
        fix=True,
    )

    assert len(valid) == 1
    assert len(invalid) == 1
    assert len(unreadable) == 1

    rewritten = [
        json.loads(line)
        for line in responses_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rewritten == [valid_record]

    failed_lines = [
        json.loads(line)
        for line in failed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(failed_lines) == 1
    assert failed_lines[0]["query_id"] == "q2"
