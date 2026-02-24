"""Call teacher LLM for synthetic queries and store rich request/response artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import NAMESPACE_URL, uuid5

from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed


def _stable_uuid(kind: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"fitsense:{kind}:{value}"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_latest_queries(raw_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    latest_path = raw_root / "synthetic_queries" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing queries latest pointer: {latest_path}. Run generate_synthetic_queries.py first."
        )

    latest_meta = json.loads(latest_path.read_text(encoding="utf-8"))
    queries_path = Path(latest_meta["run_dir"]) / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    queries: list[dict[str, Any]] = []
    with queries_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    return latest_meta, queries


def _build_system_prompt() -> str:
    return (
        "You are a senior fitness coach AI. Produce safe, structured, concise guidance. "
        "Respect medical constraints and avoid unsafe exercise recommendations."
    )


def _mock_teacher_response(query: dict[str, Any]) -> str:
    prompt_type = str(query.get("prompt_type", "unknown"))
    goal = str(query.get("slice_tags", {}).get("goal_type", "general_fitness"))
    activity = str(query.get("slice_tags", {}).get("activity_level", "moderate"))
    constraints = query.get("expected_safety_constraints", [])
    constraints_text = "; ".join(constraints[:3]) if constraints else "respect safety constraints"

    if prompt_type == "plan_creation":
        return (
            f"Weekly Plan (goal: {goal}, activity: {activity}): 4 training days, 2 active recovery days, 1 rest day. "
            "Main lifts at RIR 2-3, accessory work at RIR 1-2, and progressive overload of 2-5% weekly. "
            f"Safety: {constraints_text}."
        )
    if prompt_type == "plan_modification":
        return (
            "Plan Update: reduce total set volume by 10% for high-fatigue patterns, keep primary compound movement "
            "first, and rotate one accessory exercise to reduce overuse risk. Maintain progression only when form is stable."
        )
    if prompt_type == "safety_adjustment":
        return (
            "Safety Adjustments: remove contraindicated high-impact or high-spinal-load movements, substitute with supported "
            "variants, cap effort to RIR >= 2, and add longer rest intervals with pain-monitoring checkpoints."
        )
    if prompt_type == "progress_adaptation":
        return (
            "Adaptation Strategy: if plateau persists for 2 weeks, apply a deload week (-20% volume), then resume progressive "
            "loading with smaller increments and readiness-based set adjustments."
        )

    return "Provide safe, goal-aligned guidance with explicit progression and recovery instructions."


def _call_openai_compatible(
    query: dict[str, Any],
    cfg: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    endpoint = str(cfg.get("endpoint_url", "")).strip()
    if not endpoint:
        raise ValueError("phase4.teacher_llm.endpoint_url is required for openai_compatible provider")

    api_key_env = str(cfg.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in env var: {api_key_env}")

    payload = {
        "model": str(cfg["model_name"]),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(query["prompt_text"])},
        ],
        "temperature": float(cfg.get("temperature", 0.2)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "max_tokens": int(cfg.get("max_output_tokens", 512)),
    }

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    timeout = int(cfg.get("timeout_seconds", 30))
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        raw = json.loads(body)

    content = (
        raw.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not content:
        raise ValueError("Teacher response content is empty")

    return {
        "request_payload": payload,
        "response_text": content,
        "raw_response": raw,
    }


def _post_validate(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    has_content = len(text) > 40
    mentions_safety = any(token in text.lower() for token in ["safe", "safety", "injur", "contraind", "rir"]) 
    return {
        "has_content": has_content,
        "mentions_safety_or_load_control": mentions_safety,
        "is_valid": bool(has_content),
    }


def _detect_safety_flags(response_text: str) -> list[str]:
    flags: list[str] = []
    lowered = response_text.lower()
    if "max out" in lowered or "to failure every set" in lowered:
        flags.append("potential_overexertion_language")
    if "ignore pain" in lowered:
        flags.append("unsafe_pain_instruction")
    return flags


def _invoke_teacher_with_retry(query: dict[str, Any], cfg: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    provider = str(cfg.get("provider", "mock"))
    retries = int(cfg.get("max_retries", 3))
    backoff = float(cfg.get("retry_backoff_seconds", 1.5))

    last_err: str | None = None
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            if provider == "mock":
                response_text = _mock_teacher_response(query)
                latency_ms = int((time.perf_counter() - start) * 1000)
                return {
                    "status": "success",
                    "attempt_count": attempt,
                    "latency_ms": latency_ms,
                    "request_payload": {
                        "provider": "mock",
                        "model": str(cfg.get("model_name", "teacher-mock-v1")),
                        "prompt": str(query["prompt_text"]),
                    },
                    "raw_response": {"mock": True},
                    "response_text": response_text,
                    "error": None,
                }

            if provider == "openai_compatible":
                result = _call_openai_compatible(query=query, cfg=cfg, system_prompt=system_prompt)
                latency_ms = int((time.perf_counter() - start) * 1000)
                return {
                    "status": "success",
                    "attempt_count": attempt,
                    "latency_ms": latency_ms,
                    "request_payload": result["request_payload"],
                    "raw_response": result["raw_response"],
                    "response_text": result["response_text"],
                    "error": None,
                }

            raise ValueError(f"Unsupported provider: {provider}")

        except (ValueError, KeyError, error.URLError, error.HTTPError, TimeoutError) as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(backoff * attempt)

    return {
        "status": "failed",
        "attempt_count": retries,
        "latency_ms": 0,
        "request_payload": None,
        "raw_response": None,
        "response_text": "",
        "error": last_err or "unknown_error",
    }


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")


def call_teacher_llm(params: dict[str, Any], raw_root: Path, run_id: str | None = None) -> tuple[list[dict[str, Any]], Path]:
    cfg = dict(params["phase4"]["teacher_llm"])
    latest_queries_meta, queries = _load_latest_queries(raw_root=raw_root)

    max_queries = cfg.get("max_queries")
    if max_queries is not None:
        queries = queries[: int(max_queries)]

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    system_prompt = _build_system_prompt()
    output_dir = raw_root / "teacher_outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for query in queries:
        invoke_result = _invoke_teacher_with_retry(query=query, cfg=cfg, system_prompt=system_prompt)
        post_validation = _post_validate(invoke_result["response_text"])
        safety_flags = _detect_safety_flags(invoke_result["response_text"]) if invoke_result["response_text"] else []

        record = {
            "response_id": _stable_uuid("teacher_response", str(query["query_id"])),
            "query_id": query["query_id"],
            "scenario_id": query["scenario_id"],
            "user_id": query["user_id"],
            "prompt_type": query["prompt_type"],
            "prompt_text": query["prompt_text"],
            "provider": cfg.get("provider", "mock"),
            "model_name": cfg.get("model_name", "teacher-mock-v1"),
            "status": invoke_result["status"],
            "attempt_count": invoke_result["attempt_count"],
            "latency_ms": invoke_result["latency_ms"],
            "request_payload": invoke_result["request_payload"],
            "response_text": invoke_result["response_text"],
            "raw_response": invoke_result["raw_response"],
            "error": invoke_result["error"],
            "safety_flags": safety_flags,
            "post_validation": post_validation,
            "source_query_run_id": latest_queries_meta["run_id"],
            "created_at": _utc_now(),
        }
        records.append(record)

    responses_jsonl = output_dir / "responses.jsonl"
    responses_csv = output_dir / "responses.csv"
    summary_json = output_dir / "summary.json"

    _write_jsonl(records=records, output_path=responses_jsonl)

    # CSV keeps nested columns serialized for simple inspection.
    import pandas as pd  # local import to keep module load lighter

    df = pd.DataFrame(records)
    for col in ["request_payload", "raw_response", "safety_flags", "post_validation"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else "")
    df.to_csv(responses_csv, index=False)

    success_count = sum(1 for r in records if r["status"] == "success")
    failed_count = len(records) - success_count

    summary = {
        "run_id": run_id,
        "run_dir": str(output_dir),
        "source_query_run_id": latest_queries_meta["run_id"],
        "provider": cfg.get("provider", "mock"),
        "model_name": cfg.get("model_name", "teacher-mock-v1"),
        "num_requests": len(records),
        "success_count": success_count,
        "failed_count": failed_count,
        "created_at": _utc_now(),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_payload = {
        "run_id": run_id,
        "run_dir": str(output_dir),
        "source_query_run_id": latest_queries_meta["run_id"],
        "num_requests": len(records),
        "provider": cfg.get("provider", "mock"),
        "model_name": cfg.get("model_name", "teacher-mock-v1"),
    }

    latest_path = raw_root / "teacher_outputs" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return records, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Call teacher LLM for synthetic queries")
    parser.add_argument("--params", default="Data-Pipeline/params.yaml", help="Path to params.yaml")
    parser.add_argument("--raw-root", default=None, help="Optional raw data root override")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    params = load_params(args.params)
    apply_global_seed(int(params["reproducibility"]["seed"]), str(params["reproducibility"]["hash_seed"]))

    raw_root = Path(args.raw_root) if args.raw_root else Path(str(params["paths"]["raw_data_dir"]))
    logger = setup_logger(
        name="fitsense.teacher_llm",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    records, out_dir = call_teacher_llm(params=params, raw_root=raw_root, run_id=args.run_id)
    success_count = sum(1 for r in records if r["status"] == "success")
    logger.info("Teacher LLM run completed: %d/%d success. Output: %s", success_count, len(records), out_dir)


if __name__ == "__main__":
    main()
