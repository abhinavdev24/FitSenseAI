from __future__ import annotations

import json


def _valid_plan_json() -> dict:
    return {
        "plan_name": "Recovered Plan",
        "days": [
            {
                "name": "FULL_1",
                "day_order": 1,
                "notes": None,
                "exercises": [
                    {
                        "exercise_name": "Squat",
                        "position": 1,
                        "notes": None,
                        "sets": [
                            {
                                "set_number": 1,
                                "target_reps": 8,
                                "target_rir": 2,
                                "rest_seconds": 120,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_extract_usage_and_parse_json_helpers(module_loader):
    module = module_loader("call_teacher.py")

    assert module._extract_usage(
        {"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}
    ) == (3, 4, 7)
    assert module._extract_usage(None) == (0, 0, 0)

    fenced = '```json\n{"ok": true}\n```'
    assert module._try_parse_json(fenced) == {"ok": True}

    think_then_json = '<think>analysis</think>{"value": 10}'
    assert module._try_parse_json(think_then_json) == {"value": 10}

    truncated_think = "<think>never closes"
    assert module._try_parse_json(truncated_think) is None


def test_resolve_cfg_applies_overrides(module_loader):
    module = module_loader("call_teacher.py")
    params = {"teacher_llm": {"model_name": "base", "temperature": 0.2}}
    cli = {"model_name": "override", "temperature": None, "max_workers": 2}

    cfg = module._resolve_cfg(params, cli)
    assert cfg["model_name"] == "override"
    assert cfg["temperature"] == 0.2
    assert cfg["max_workers"] == 2


def test_input_token_rate_limiter_records_tokens(module_loader):
    module = module_loader("call_teacher.py")
    limiter = module.InputTokenRateLimiter(tokens_per_minute=100)

    assert limiter.tokens_used_in_window() == 0
    assert limiter.wait_if_needed(estimated_input_tokens=10) == 0
    limiter.record(25)
    assert limiter.tokens_used_in_window() >= 25


def test_load_existing_responses_promotes_failed_records(
    tmp_path,
    module_loader,
    monkeypatch,
):
    module = module_loader("call_teacher.py")

    responses_path = tmp_path / "responses.jsonl"
    failed_path = tmp_path / "failed_responses.jsonl"
    responses_path.write_text("", encoding="utf-8")

    valid_existing = {
        "response_id": "ok-1",
        "query_id": "q-ok",
        "status": "success",
        "response_json": _valid_plan_json(),
        "raw_response": {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        },
    }

    monkeypatch.setattr(
        module,
        "validate_jsonl_file",
        lambda responses_path, failed_path, fix: ([valid_existing], [], []),
    )

    rescued = {
        "response_id": "bad-1",
        "query_id": "q-recover",
        "user_id": "u1",
        "prompt_type": "plan_creation",
        "request_payload": {"messages": [{"role": "user", "content": "prompt"}]},
        "response_text": json.dumps(_valid_plan_json()),
        "status": "failed",
    }
    failed_path.write_text(json.dumps(rescued) + "\n", encoding="utf-8")

    valid_records, completed_ids, retry = module._load_existing_responses(
        responses_path=responses_path,
        failed_path=failed_path,
    )

    assert "q-ok" in completed_ids
    assert "q-recover" in completed_ids
    assert retry == []
    assert len(valid_records) == 2
    assert failed_path.read_text(encoding="utf-8").strip() == ""
