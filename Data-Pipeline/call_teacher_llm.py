"""Call teacher LLM for synthetic queries and store rich request/response artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from groq import APIConnectionError, APIStatusError, Groq
from tqdm import tqdm
from common.config import load_params
from common.logging_utils import setup_logger
from common.reproducibility import apply_global_seed
from common.prompt_builder import build_system_prompt


# Action types that benefit from extended reasoning (complex planning / safety-critical).
# All other types use standard inference (thinking disabled) for lower latency and cost.
THINKING_ENABLED_TYPES: frozenset[str] = frozenset(
    {"plan_creation", "safety_adjustment", "progress_adaptation"}
)


class NonRetriableTeacherError(Exception):
    """Raised for configuration/auth/provider issues that should fail fast."""


class TruncatedResponseError(ValueError):
    """Raised when the model returned HTTP 200 but the output was cut off mid-generation.

    This covers two observable symptoms of hitting the token limit:
      - An unclosed ``<think>`` block (``</think>`` never appeared in the stream).
      - A ``json.JSONDecodeError`` after the thinking block was stripped, meaning the
        actual JSON payload was truncated before the closing brace.

    Both cases are retriable: the retry loop will escalate the token budget so the
    model has enough room to finish its output.
    """


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


def _load_env_file_if_present(repo_root: Path | None = None) -> None:
    """Load KEY=VALUE lines from a local .env file without overriding existing env vars.

    By default, the function searches in common repository locations (first match wins):
    - <repo_root>/.env.local
    - <repo_root>/.env
    - <repo_root>/Data-Pipeline/.env.local
    - <repo_root>/Data-Pipeline/.env
    - <repo_root>/Data-Pipeline/scripts/.env.local
    - <repo_root>/Data-Pipeline/scripts/.env

    The optional `repo_root` parameter is primarily used in tests to point the loader
    at a temporary directory instead of the real workspace.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / ".env.local",
        repo_root / ".env",
        repo_root / "Data-Pipeline" / ".env.local",
        repo_root / "Data-Pipeline" / ".env",
        repo_root / "Data-Pipeline" / "scripts" / ".env.local",
        repo_root / "Data-Pipeline" / "scripts" / ".env",
    ]

    for env_path in candidates:
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

        return


def _extract_thinking(response_text: str) -> tuple[str | None, str]:
    """Extract thinking content from <think>...</think> blocks in response text.

    Returns:
        A tuple of (thinking_text, remaining_text) where thinking_text is the content
        between <think> and </think> tags (or None if no tags found), and remaining_text
        is the response with the thinking block removed and stripped.
    """
    open_tag = "<think>"
    close_tag = "</think>"

    open_idx = response_text.find(open_tag)
    if open_idx == -1:
        return None, response_text.strip()

    close_idx = response_text.find(close_tag, open_idx + len(open_tag))
    if close_idx == -1:
        # The closing tag is absent — the model was cut off inside the thinking block.
        # Raise so the retry loop can escalate the token budget.
        raise TruncatedResponseError(
            "Response truncated: <think> opened but </think> never closed. "
            "Output token limit was too low to finish the reasoning block."
        )

    thinking_text = response_text[open_idx + len(open_tag) : close_idx]
    remaining = response_text[:open_idx] + response_text[close_idx + len(close_tag) :]
    remaining = remaining.strip()

    return thinking_text if thinking_text else None, remaining


def _call_groq(
    query: dict[str, Any],
    cfg: dict[str, Any],
    system_prompt: str,
    logger: Any = None,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    api_key_env = str(cfg.get("api_key_env", "GROQ_API_KEY"))
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise NonRetriableTeacherError(
            f"Missing API key in env var: {api_key_env}. Set it in the environment or .env."
        )

    user_content = str(query["prompt_text"])
    model = str(cfg["model_name"])
    timeout = int(cfg.get("timeout_seconds", 30))

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Resolve token limits from the per-mode sub-block in params.yaml, falling back
    # to the top-level max_output_tokens if the sub-block or key is absent.
    # Qwen3 thinking mode also requires temperature=1 — enforced unconditionally here.
    fallback_max_tokens = int(cfg.get("max_output_tokens", 2048))
    if enable_thinking:
        thinking_cfg = cfg.get("thinking", {})
        max_output_tokens = int(
            thinking_cfg.get("max_output_tokens", fallback_max_tokens)
        )
        budget_tokens = int(
            thinking_cfg.get("thinking_budget_tokens", max_output_tokens // 3)
        )
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 1.0,
            "top_p": float(cfg.get("top_p", 1.0)),
            "max_completion_tokens": max_output_tokens,
            "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
        }
    else:
        non_thinking_cfg = cfg.get("non_thinking", {})
        max_output_tokens = int(
            non_thinking_cfg.get("max_output_tokens", fallback_max_tokens)
        )
        create_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": float(cfg.get("temperature", 0.2)),
            "top_p": float(cfg.get("top_p", 1.0)),
            "max_completion_tokens": max_output_tokens,
        }

    if logger:
        thinking_info = f"budget={budget_tokens}" if enable_thinking else "off"
        logger.info(
            "  -> Groq API | model=%s | prompt_type=%s | thinking=%s | max_tokens=%d | timeout=%ds | prompt_chars=%d",
            model,
            query.get("prompt_type", "?"),
            thinking_info,
            max_output_tokens,
            timeout,
            len(user_content),
        )

    client = Groq(api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(**create_kwargs)

    if logger:
        usage = response.usage
        logger.info(
            "  <- tokens: prompt=%s completion=%s total=%s",
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
            usage.total_tokens if usage else "?",
        )

    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise ValueError("Teacher response content is empty")

    # Strip markdown fences if model wraps response despite instructions
    if content.startswith("```"):
        fence_lines = content.split("\n")
        content = "\n".join(
            line for line in fence_lines if not line.strip().startswith("```")
        ).strip()

    # Strip <think>...</think> blocks that Qwen3 may emit before the JSON
    _, content = _extract_thinking(content)

    try:
        response_json = json.loads(content)
    except json.JSONDecodeError as exc:
        if logger:
            logger.error(
                "  !! JSON parse failed (likely truncated). Raw content (first 500 chars): %s",
                content[:500],
            )
        err = TruncatedResponseError(f"Teacher response is not valid JSON: {exc}")
        err._raw_response_text = content  # type: ignore[attr-defined]
        err._request_payload = create_kwargs  # type: ignore[attr-defined]
        raise err from exc

    if logger:
        logger.info("  <- JSON parsed OK | keys=%s", list(response_json.keys()))

    return {
        "request_payload": create_kwargs,
        "response_text": content,
        "response_json": response_json,
        "raw_response": response.model_dump(),
    }


def _post_validate(
    response_text: str, response_json: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate that the teacher response is a well-formed JSON coaching response."""
    if not response_json:
        return {
            "has_message_field": False,
            "has_data_field": False,
            "mentions_safety_or_load_control": False,
            "is_valid": False,
        }
    has_message = (
        isinstance(response_json.get("message"), str)
        and len(response_json["message"]) > 10
    )
    has_data_key = (
        "data" in response_json
    )  # null is valid — check key presence, not truthiness
    mentions_safety = any(
        token in response_json.get("message", "").lower()
        for token in ["safe", "safety", "injur", "contraind", "rir"]
    )
    return {
        "has_message_field": has_message,
        "has_data_field": has_data_key,
        "mentions_safety_or_load_control": mentions_safety,
        "is_valid": has_message and has_data_key,
    }


def _detect_safety_flags(
    response_text: str, response_json: dict[str, Any] | None = None
) -> list[str]:
    """Scan the human-readable message field for unsafe coaching language."""
    scan_text = response_text
    if response_json and isinstance(response_json.get("message"), str):
        scan_text = response_json["message"]
    lowered = scan_text.lower()
    flags: list[str] = []
    if "max out" in lowered or "to failure every set" in lowered:
        flags.append("potential_overexertion_language")
    if "ignore pain" in lowered:
        flags.append("unsafe_pain_instruction")
    return flags


def _build_escalated_cfg(
    cfg: dict[str, Any], originally_thinking: bool
) -> dict[str, Any]:
    """Return a shallow-copied cfg with token limits escalated for a truncation retry.

    Rules
    -----
    - Non-thinking request that was truncated → retry WITH thinking enabled, using the
      standard thinking token limits from ``cfg["thinking"]``.
    - Thinking request that was truncated → retry with thinking still enabled but
      ``max_output_tokens`` and ``thinking_budget_tokens`` both raised by 50 %.
    """
    escalated = dict(cfg)
    thinking_cfg = dict(cfg.get("thinking", {}))
    fallback_max = int(cfg.get("max_output_tokens", 2048))

    if not originally_thinking:
        # Promote to thinking mode using the standard thinking limits unchanged.
        escalated["thinking"] = thinking_cfg
    else:
        # Already thinking — bump both caps by 50 %.
        base_max = int(thinking_cfg.get("max_output_tokens", fallback_max))
        base_budget = int(thinking_cfg.get("thinking_budget_tokens", base_max // 3))
        thinking_cfg["max_output_tokens"] = int(base_max * 1.5)
        thinking_cfg["thinking_budget_tokens"] = int(base_budget * 1.5)
        escalated["thinking"] = thinking_cfg

    return escalated


def _invoke_teacher_with_retry(
    query: dict[str, Any],
    cfg: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    provider = str(cfg.get("provider", "groq"))
    if provider != "groq":
        raise NonRetriableTeacherError(
            f"Unsupported provider: {provider!r}. Only 'groq' is supported."
        )

    retries = int(cfg.get("max_retries", 3))
    backoff = float(cfg.get("retry_backoff_seconds", 1.5))

    # Track whether thinking was originally requested for this prompt type, and
    # whether the current attempt is running with thinking enabled (may be escalated).
    originally_thinking = query.get("prompt_type") in THINKING_ENABLED_TYPES
    enable_thinking = originally_thinking
    active_cfg = cfg  # cfg used for the current attempt; may be escalated on truncation

    last_err: str | None = None
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            result = _call_groq(
                query=query,
                cfg=active_cfg,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "status": "success",
                "attempt_count": attempt,
                "latency_ms": latency_ms,
                "request_payload": result["request_payload"],
                "raw_response": result["raw_response"],
                "response_text": result["response_text"],
                "response_json": result["response_json"],
                "error": None,
            }

        except NonRetriableTeacherError as exc:
            return {
                "status": "failed",
                "attempt_count": attempt,
                "latency_ms": 0,
                "request_payload": None,
                "raw_response": None,
                "response_text": "",
                "response_json": None,
                "error": str(exc),
            }

        except TruncatedResponseError as exc:
            # HTTP 200 but output was cut off (unclosed <think> or invalid JSON).
            # Escalate token limits and force thinking on for the next attempt.
            last_err = str(exc)
            if attempt < retries:
                active_cfg = _build_escalated_cfg(
                    active_cfg, originally_thinking=enable_thinking
                )
                enable_thinking = True  # always thinking after first truncation
                time.sleep(backoff * attempt)

        except (ValueError, KeyError) as exc:
            # Deterministic data/schema issues — not caused by truncation, not retriable.
            return {
                "status": "failed",
                "attempt_count": attempt,
                "latency_ms": 0,
                "request_payload": getattr(exc, "_request_payload", None),
                "raw_response": None,
                "response_text": getattr(exc, "_raw_response_text", ""),
                "response_json": None,
                "error": str(exc),
            }
        except APIStatusError as exc:
            last_err = str(exc)
            if exc.status_code < 500:
                # 4xx errors (bad request, auth) are non-retriable
                return {
                    "status": "failed",
                    "attempt_count": attempt,
                    "latency_ms": 0,
                    "request_payload": None,
                    "raw_response": None,
                    "response_text": "",
                    "response_json": None,
                    "error": last_err,
                }
            if attempt < retries:
                time.sleep(backoff * attempt)
        except APIConnectionError as exc:
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
        "response_json": None,
        "error": last_err or "unknown_error",
    }


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")


def call_teacher_llm(
    params: dict[str, Any], raw_root: Path, run_id: str | None = None
) -> tuple[list[dict[str, Any]], Path]:
    cfg = dict(params["phase4"]["teacher_llm"])
    latest_queries_meta, queries = _load_latest_queries(raw_root=raw_root)

    max_queries = cfg.get("max_queries")
    if max_queries is not None:
        queries = queries[: int(max_queries)]

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output_dir = raw_root / "teacher_outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    request_delay = float(cfg.get("request_delay_seconds", 10))
    records: list[dict[str, Any]] = []

    passed = 0
    failed = 0
    progress = tqdm(
        total=len(queries),
        desc="Teacher LLM",
        unit="req",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        dynamic_ncols=True,
    )
    progress.set_postfix(passed=passed, failed=failed)

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(request_delay)
        system_prompt = build_system_prompt(action=query["prompt_type"])
        invoke_result = _invoke_teacher_with_retry(
            query=query,
            cfg=cfg,
            system_prompt=system_prompt,
        )

        cleaned_response_text = invoke_result["response_text"]

        post_validation = _post_validate(
            cleaned_response_text,
            response_json=invoke_result.get("response_json"),
        )
        safety_flags = (
            _detect_safety_flags(
                cleaned_response_text,
                response_json=invoke_result.get("response_json"),
            )
            if cleaned_response_text
            else []
        )

        record = {
            "response_id": _stable_uuid("teacher_response", str(query["query_id"])),
            "query_id": query["query_id"],
            "scenario_id": query["scenario_id"],
            "user_id": query["user_id"],
            "prompt_type": query["prompt_type"],
            "system_prompt": system_prompt,
            "prompt_text": query["prompt_text"],
            "provider": cfg.get("provider", "mock"),
            "model_name": cfg.get("model_name", "teacher-mock-v1"),
            "status": invoke_result["status"],
            "attempt_count": invoke_result["attempt_count"],
            "latency_ms": invoke_result["latency_ms"],
            "request_payload": invoke_result["request_payload"],
            "response_text": cleaned_response_text,
            "response_json": invoke_result.get("response_json"),
            "raw_response": invoke_result["raw_response"],
            "error": invoke_result["error"],
            "safety_flags": safety_flags,
            "post_validation": post_validation,
            "source_query_run_id": latest_queries_meta["run_id"],
            "created_at": _utc_now(),
        }
        records.append(record)

        if invoke_result["status"] == "success":
            passed += 1
        else:
            failed += 1
        progress.set_postfix(passed=passed, failed=failed)
        progress.update(1)

    progress.close()

    responses_jsonl = output_dir / "responses.jsonl"
    responses_csv = output_dir / "responses.csv"
    summary_json = output_dir / "summary.json"
    failed_responses_jsonl = output_dir / "failed_responses.jsonl"

    success_records = [r for r in records if r["status"] == "success"]
    _write_jsonl(records=success_records, output_path=responses_jsonl)

    failed_records = [r for r in records if r["status"] != "success"]
    _write_jsonl(records=failed_records, output_path=failed_responses_jsonl)

    # CSV keeps nested columns serialized for simple inspection.
    import pandas as pd  # local import to keep module load lighter

    df = pd.DataFrame(records)
    for col in [
        "request_payload",
        "raw_response",
        "safety_flags",
        "post_validation",
        "response_json",
    ]:
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
        "failed_responses_file": str(failed_responses_jsonl),
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
    parser = argparse.ArgumentParser(
        description="Call teacher LLM for synthetic queries"
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument(
        "--raw-root", default=None, help="Optional raw data root override"
    )
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    _load_env_file_if_present()

    params = load_params(args.params)
    apply_global_seed(
        int(params["reproducibility"]["seed"]),
        str(params["reproducibility"]["hash_seed"]),
    )

    raw_root = (
        Path(args.raw_root)
        if args.raw_root
        else Path(str(params["paths"]["raw_data_dir"]))
    )
    logger = setup_logger(
        name="fitsense.teacher_llm",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    records, out_dir = call_teacher_llm(
        params=params, raw_root=raw_root, run_id=args.run_id
    )
    success_count = sum(1 for r in records if r["status"] == "success")
    logger.info(
        "Teacher LLM run completed: %d/%d success. Output: %s",
        success_count,
        len(records),
        out_dir,
    )


if __name__ == "__main__":
    main()
