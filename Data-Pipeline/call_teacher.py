"""
call_teacher.py
===============
Calls Groq or OpenRouter (qwen3-32b) for each query in the latest
generate_queries run.

Features
--------
- Provider-agnostic: set teacher_llm.provider to "groq" or "openrouter"
  in params.yaml.  Both use the openai-compatible SDK.
- Reads params.yaml for all configuration.
- Loads API keys from .env / .env.local:
    GROQ_API_KEY       — for provider: groq
    OPENROUTER_API_KEY — for provider: openrouter
- Thinking is ALWAYS enabled.
- System prompt loaded per prompt_type from prompts/<prompt_type>.md.
- Resumes automatically: output lives at a fixed path tied to the query run_id.
  On resume, existing responses.jsonl is validated with validate.py:
    - Records that fail structural validation are moved to failed_responses.jsonl
      and re-queued for calling.
    - Only structurally valid, successful records are treated as completed.
- Flushes responses.jsonl and failed_responses.jsonl after every request.
- Progress bar with live success/failed/token counts.
- Logs only startup info and a final summary.

Output (paths.responses_root/<source_run_id>/)
----------------------------------------------
  responses.jsonl          — valid completed records
  responses.csv            — same, flattened for spreadsheet review
  failed_responses.jsonl   — API failures + JSON parse failures + validation failures
  summary.json             — aggregate stats written at end of run

Usage
-----
  python call_teacher.py
  python call_teacher.py --dry-run
  python call_teacher.py --max-queries 5
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

# Adjust this import path to match your project layout.
from common.logging_utils import setup_logger
from validate import validate_jsonl_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDER = "groq"  # overridden at runtime from params.yaml
PROMPTS_DIR = Path(__file__).parent / "prompts"

RESPONSE_FIELDS = [
    "response_id",
    "query_id",
    "user_id",
    "prompt_type",
    "provider",
    "model_name",
    "status",
    "attempt_count",
    "request_payload",
    "response_text",
    "response_json",
    "raw_response",
    "source_query_run_id",
    "created_at",
]

# ---------------------------------------------------------------------------
# Module-level logger placeholder — replaced in main() after params load
# ---------------------------------------------------------------------------

log = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Token usage helpers
# ---------------------------------------------------------------------------


def _extract_usage(raw_response: dict | None) -> tuple[int, int, int]:
    """
    Extract (prompt_tokens, completion_tokens, total_tokens) from a Groq
    raw_response dict.  Groq always populates usage on successful calls.
    Returns (0, 0, 0) if unavailable (e.g. failed requests).
    """
    if not raw_response or not isinstance(raw_response, dict):
        return 0, 0, 0
    usage = raw_response.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    total = int(usage.get("total_tokens", 0))
    return prompt, completion, total


class InputTokenRateLimiter:
    """
    Thread-safe rolling 60-second input-token rate limiter.

    Before each request call .wait_if_needed(estimated_input_tokens).
    After the request resolves call .record(actual_prompt_tokens).
    All methods are protected by a threading.Lock so multiple workers
    can share a single instance safely.
    """

    WINDOW_SECONDS = 60

    def __init__(self, tokens_per_minute: int) -> None:
        self.limit = tokens_per_minute
        self._window: collections.deque[tuple[float, int]] = collections.deque()
        self._lock = threading.Lock()

    def _evict_old(self, now: float) -> None:
        """Must be called with self._lock held."""
        cutoff = now - self.WINDOW_SECONDS
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def tokens_used_in_window(self) -> int:
        with self._lock:
            now = time.monotonic()
            self._evict_old(now)
            return sum(t for _, t in self._window)

    def wait_if_needed(self, estimated_input_tokens: int) -> float:
        """
        Block until adding estimated_input_tokens won't exceed the limit.
        Returns the number of seconds slept (0 if no wait needed).
        Safe to call from multiple threads concurrently.
        """
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict_old(now)
                if not self._window:
                    # Window empty — first request, no wait needed
                    break
                used = sum(t for _, t in self._window)
                if used + estimated_input_tokens <= self.limit:
                    break
                # Need to wait — calculate sleep outside the lock
                oldest_ts = self._window[0][0]
                sleep_for = (oldest_ts + self.WINDOW_SECONDS) - now + 0.05

            if sleep_for > 0:
                time.sleep(sleep_for)
                slept += sleep_for

        return slept

    def record(self, prompt_tokens: int) -> None:
        """Record that prompt_tokens were consumed right now."""
        with self._lock:
            self._window.append((time.monotonic(), prompt_tokens))


# ---------------------------------------------------------------------------
# Env loader
# ---------------------------------------------------------------------------


def _load_env() -> None:
    root = Path(__file__).parent
    for env_file in [".env", ".env.local"]:
        p = root / env_file
        if p.exists():
            load_dotenv(p, override=True)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def _load_params(params_path: Path) -> dict:
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_cfg(params: dict, cli_overrides: dict) -> dict:
    cfg = dict(params.get("teacher_llm", {}))
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

_prompt_cache: dict[str, str] = {}


def _load_system_prompt(prompt_type: str) -> str:
    if prompt_type in _prompt_cache:
        return _prompt_cache[prompt_type]
    prompt_file = PROMPTS_DIR / f"{prompt_type}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {prompt_file}\n"
            f"Expected '{prompt_type}.md' in {PROMPTS_DIR}"
        )
    content = prompt_file.read_text(encoding="utf-8").strip()
    _prompt_cache[prompt_type] = content
    return content


# ---------------------------------------------------------------------------
# Queries loader
# ---------------------------------------------------------------------------


def _load_latest_queries(raw_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    latest_path = raw_root / "synthetic_queries" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing queries latest pointer: {latest_path}. "
            "Run generate_synthetic_queries.py first."
        )
    latest_meta = json.loads(latest_path.read_text(encoding="utf-8"))
    queries_path = Path(latest_meta["run_dir"]) / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    queries: list[dict[str, Any]] = []
    with queries_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return latest_meta, queries


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _load_existing_responses(
    responses_path: Path,
    failed_path: Path,
) -> tuple[list[dict], set[str], list[dict]]:
    """
    Read existing responses.jsonl and failed_responses.jsonl on resume.

    The flow:
    1. Validate responses.jsonl — keep valid, move newly-invalid to failed.
    2. Re-validate failed_responses.jsonl with the (possibly updated) validator.
       Records that now pass validation are promoted back: added to
       responses.jsonl and removed from failed_responses.jsonl.
    3. Remaining failed records whose query_id is NOT in the valid set
       are returned as retry candidates.

    Returns (valid_records, completed_query_ids, retry_queries).
    retry_queries are dicts with at least "query_id", "user_id",
    "prompt_type", "prompt_text" so they can be re-sent to the LLM.
    """
    if not responses_path.exists():
        return [], set(), []

    # Step 1: validate responses.jsonl (rewrites it, moves bad to failed)
    valid_records, invalid_records, _ = validate_jsonl_file(
        responses_path=responses_path,
        failed_path=failed_path,
        fix=True,
    )

    completed: set[str] = {r["query_id"] for r in valid_records}

    # Step 2: re-validate every record in failed_responses.jsonl
    if not failed_path.exists():
        return valid_records, completed, []

    failed_records: list[dict] = []
    with failed_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    failed_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not failed_records:
        return valid_records, completed, []

    # Re-parse response_text for failed records that have one — the updated
    # _try_parse_json may now extract valid JSON that was previously rejected
    # (e.g. markdown fences after think blocks).
    promoted: list[dict] = []
    still_failed: list[dict] = []

    for rec in failed_records:
        qid = rec.get("query_id", "")

        # Already in valid set (duplicate) — drop from failed
        if qid in completed:
            continue

        # Try to rescue: re-parse response_text and re-validate
        rescued = False
        resp_text = rec.get("response_text") or ""
        if resp_text:
            parsed = _try_parse_json(resp_text)
            if parsed is not None:
                from validate import validate_response_json

                vr = validate_response_json(parsed)
                if vr.ok:
                    # Promote: update the record in-place
                    rec["response_json"] = parsed
                    rec["status"] = "success"
                    rec.pop("_validation_error", None)
                    promoted.append(rec)
                    completed.add(qid)
                    rescued = True

        if not rescued:
            still_failed.append(rec)

    # Apply promotions: append promoted to responses.jsonl
    if promoted:
        with responses_path.open("a", encoding="utf-8") as fh:
            for rec in promoted:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        valid_records.extend(promoted)

    # Rewrite failed_responses.jsonl with only the still-failed records
    with failed_path.open("w", encoding="utf-8") as fh:
        for rec in still_failed:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Step 3: build retry candidates from still-failed records whose
    # query_id is not yet completed.  We reconstruct the minimal query
    # dict that process_query() expects.
    retry_queries: list[dict] = []
    for rec in still_failed:
        qid = rec.get("query_id", "")
        if qid not in completed:
            retry_queries.append(
                {
                    "query_id": qid,
                    "user_id": rec.get("user_id", "unknown"),
                    "prompt_type": rec.get("prompt_type", "unknown"),
                    "prompt_text": (rec.get("request_payload") or {})
                    .get("messages", [{}])[-1]
                    .get("content", ""),
                }
            )

    return valid_records, completed, retry_queries


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> Any:
    """
    Strip <think>…</think> blocks, strip ```json fences, parse JSON.
    Returns parsed object or None on failure.
    """
    # 1. Handle <think> blocks first — before fence stripping.
    #    If <think> opens but never closes, the response was truncated
    #    during the thinking phase — no usable JSON exists.
    if "<think>" in text:
        if "</think>" not in text:
            return None  # thinking phase exhausted the token budget
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    else:
        cleaned = text.strip()

    # 2. Strip markdown fences (```json ... ```) AFTER think removal.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Core API caller
# ---------------------------------------------------------------------------


def _call_llm(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: float,
    provider: str,
) -> tuple[str, dict, dict]:
    """
    Fire one completion via the OpenAI-compatible client.
    Works for both Groq (base_url=https://api.groq.com/openai/v1)
    and OpenRouter (base_url=https://openrouter.ai/api/v1).
    Returns (response_text, raw_response_dict, request_payload_dict).
    """
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = client.chat.completions.create(**payload, timeout=timeout_seconds)
    text = response.choices[0].message.content or ""
    raw = response.model_dump()
    # Normalise usage — OpenRouter mirrors Groq's usage shape
    return text, raw, payload


# ---------------------------------------------------------------------------
# Per-query processor
# ---------------------------------------------------------------------------


def process_query(
    query: dict,
    client: OpenAI,
    model_name: str,
    provider: str,
    source_run_id: str,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    rate_limiter: InputTokenRateLimiter,
) -> tuple[dict, int, int, int]:
    """
    Process one query.
    Returns (record, prompt_tokens, completion_tokens, total_tokens).
    Token counts come from the provider's usage field — no estimation needed.
    Works for both Groq and OpenRouter.
    """
    query_id = query.get("query_id", "unknown")
    user_id = query.get("user_id", "unknown")
    prompt_type = query.get("prompt_type", "unknown")
    system_prompt = _load_system_prompt(prompt_type)
    user_prompt = query.get("prompt_text", "")

    status = "pending"
    attempt_count = 0
    response_text = None
    response_json = None
    raw_response = None
    request_payload = None
    error_detail = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # Use last known prompt size as estimate; 0 on first call (no wait)
    estimated_input = rate_limiter.tokens_used_in_window() and prompt_tokens or 0

    for attempt in range(1, max_retries + 1):
        attempt_count = attempt
        try:
            # Block here if we are close to the input token rate limit
            rate_limiter.wait_if_needed(estimated_input)

            response_text, raw_response, request_payload = _call_llm(
                client,
                model_name,
                system_prompt,
                user_prompt,
                temperature,
                max_output_tokens,
                timeout_seconds,
                provider,
            )

            # Record actual tokens consumed into the rolling window
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(
                raw_response
            )
            rate_limiter.record(prompt_tokens)
            estimated_input = prompt_tokens  # use actual for any retry

            response_json = _try_parse_json(response_text)
            status = "json_parse_failed" if response_json is None else "success"
            break

        except RateLimitError as e:
            wait = retry_backoff_seconds * attempt
            error_detail = str(e)
            if attempt < max_retries:
                time.sleep(wait)

        except (APIStatusError, APIConnectionError) as e:
            error_detail = str(e)
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds)

        except Exception as e:
            error_detail = traceback.format_exc()
            break  # non-retryable

    if status == "pending":
        status = "failed"

    record = {
        "response_id": str(uuid.uuid4()),
        "query_id": query_id,
        "user_id": user_id,
        "prompt_type": prompt_type,
        "provider": provider,
        "model_name": model_name,
        "status": status,
        "attempt_count": attempt_count,
        "request_payload": request_payload,
        "response_text": response_text,
        "response_json": response_json,
        "raw_response": raw_response if raw_response else {"error": error_detail},
        "source_query_run_id": source_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return record, prompt_tokens, completion_tokens, total_tokens


# ---------------------------------------------------------------------------
# Incremental writers
# ---------------------------------------------------------------------------


def _append_jsonl(record: dict, path: Path) -> None:
    """Append a single record to a JSONL file (create if missing)."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _remove_query_from_failed(query_id: str, failed_path: Path) -> None:
    """Remove all records matching query_id from failed_responses.jsonl.

    Reads the entire file, filters out matching records, and rewrites it.
    Called when a previously-failed query succeeds on retry.
    """
    if not failed_path.exists():
        return
    kept: list[str] = []
    with failed_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                if rec.get("query_id") == query_id:
                    continue  # drop this record
            except json.JSONDecodeError:
                pass
            kept.append(stripped)
    with failed_path.open("w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")


def _rewrite_csv(records: list[dict], path: Path) -> None:
    """Rewrite the entire CSV — called after each record to keep it current."""
    if not records:
        path.write_text("", encoding="utf-8")
        return
    _CSV_DICT_FIELDS = {"request_payload", "response_json", "raw_response"}
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESPONSE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            for field in _CSV_DICT_FIELDS:
                if row.get(field) is not None:
                    row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)


def _write_summary(
    records: list[dict],
    source_run_id: str,
    elapsed: float,
    cfg: dict,
    path: Path,
    provider: str = "groq",
    sess_prompt: int = 0,
    sess_completion: int = 0,
    sess_total: int = 0,
) -> dict:
    total = len(records)
    success = sum(1 for r in records if r["status"] == "success")
    json_failed = sum(1 for r in records if r["status"] == "json_parse_failed")
    api_failed = sum(1 for r in records if r["status"] == "failed")

    # Aggregate token usage across all records from raw_response.usage
    all_prompt = sum(_extract_usage(r.get("raw_response"))[0] for r in records)
    all_completion = sum(_extract_usage(r.get("raw_response"))[1] for r in records)
    all_total = sum(_extract_usage(r.get("raw_response"))[2] for r in records)

    by_prompt_type: dict[str, dict] = {}
    for r in records:
        pt = r["prompt_type"]
        if pt not in by_prompt_type:
            by_prompt_type[pt] = {
                "total": 0,
                "success": 0,
                "json_parse_failed": 0,
                "failed": 0,
            }
        by_prompt_type[pt]["total"] += 1
        by_prompt_type[pt][r["status"]] = by_prompt_type[pt].get(r["status"], 0) + 1

    summary = {
        "source_query_run_id": source_run_id,
        "model_name": cfg.get("model_name"),
        "provider": provider,
        "thinking": True,
        "params": {
            "temperature": cfg.get("temperature"),
            "max_output_tokens": cfg.get("max_output_tokens"),
            "timeout_seconds": cfg.get("timeout_seconds"),
            "max_workers": cfg.get("max_workers"),
            "max_queries": cfg.get("max_queries"),
            "max_retries": cfg.get("max_retries"),
            "retry_backoff_seconds": cfg.get("retry_backoff_seconds"),
            "request_delay_seconds": cfg.get("request_delay_seconds"),
            "input_tokens_per_minute": cfg.get("input_tokens_per_minute"),
        },
        "total_queries": total,
        "success": success,
        "json_parse_failed": json_failed,
        "api_failed": api_failed,
        "success_rate": round(success / total, 4) if total else 0,
        "by_prompt_type": by_prompt_type,
        # token counts across ALL records (including pre-existing on resume)
        "tokens": {
            "prompt_tokens": all_prompt,
            "completion_tokens": all_completion,
            "total_tokens": all_total,
        },
        # token counts for this session only (new requests this run)
        "session_tokens": {
            "prompt_tokens": sess_prompt,
            "completion_tokens": sess_completion,
            "total_tokens": sess_total,
        },
        "elapsed_seconds": round(elapsed, 2),
        "queries_per_second": round(total / elapsed, 2) if elapsed else 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(cfg: dict, paths_cfg: dict, params: dict, dry_run: bool) -> None:
    global log

    raw_root = Path(paths_cfg["raw_data_dir"])
    responses_root = Path(paths_cfg["teacher_llm_responses_dir"])

    model_name = str(cfg["model_name"])
    temperature = float(cfg["temperature"])
    max_output_tokens = int(cfg["max_output_tokens"])
    timeout_seconds = float(cfg["timeout_seconds"])
    max_workers = int(cfg["max_workers"])
    max_queries = cfg.get("max_queries")
    max_retries = int(cfg["max_retries"])
    retry_backoff_seconds = float(cfg["retry_backoff_seconds"])
    request_delay_seconds = float(cfg["request_delay_seconds"])

    # ---- Load queries ----
    latest_meta, queries = _load_latest_queries(raw_root)
    source_run_id = latest_meta.get("run_id", Path(latest_meta["run_dir"]).name)

    if max_queries is not None:
        queries = queries[: int(max_queries)]

    if not queries:
        log.error("No queries found — aborting")
        sys.exit(1)

    # ---- Output dir: fixed path tied to source_run_id, not a timestamp ----
    out_dir = responses_root / source_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    responses_path = out_dir / "responses.jsonl"
    failed_path = out_dir / "failed_responses.jsonl"
    csv_path = out_dir / "responses.csv"
    summary_path = out_dir / "summary.json"

    # ---- Resume: validate existing records, re-queue invalid ones ----
    existing_records, completed_ids, retry_queries = _load_existing_responses(
        responses_path=responses_path,
        failed_path=failed_path,
    )

    # Pending = new queries not yet completed + failed queries to retry
    pending_queries = [q for q in queries if q.get("query_id") not in completed_ids]

    # Merge retry queries (from failed_responses.jsonl) that aren't already
    # covered by the pending list from the original query set.
    pending_qids = {q["query_id"] for q in pending_queries}
    retries_added = 0
    for rq in retry_queries:
        if rq["query_id"] not in pending_qids and rq.get("prompt_text"):
            pending_queries.append(rq)
            pending_qids.add(rq["query_id"])
            retries_added += 1

    log.info("=== call_teacher_llm ===")
    log.info("  source_run_id          : %s", source_run_id)
    log.info("  output dir             : %s", out_dir)
    log.info("  total queries          : %d", len(queries))
    log.info("  already completed      : %d", len(completed_ids))
    log.info("  retrying from failed   : %d", retries_added)
    log.info("  pending                : %d", len(pending_queries))
    log.info(
        "  model                  : %s / %s", cfg.get("provider", "groq"), model_name
    )
    log.info("  thinking               : always on")
    log.info("  temperature            : %s", temperature)
    log.info("  max_output_tokens      : %d", max_output_tokens)
    log.info("  timeout_seconds        : %s", timeout_seconds)
    log.info(
        "  max_workers            : %d  (%s)",
        max_workers,
        "sequential" if max_workers == 1 else "threaded",
    )
    log.info("  max_retries            : %d", max_retries)
    log.info("  retry_backoff_seconds  : %s", retry_backoff_seconds)
    log.info("  request_delay_seconds  : %s", request_delay_seconds)

    if not pending_queries:
        log.info("All queries already completed — nothing to do.")
        return

    # ---- Dry run ----
    if dry_run:
        q = pending_queries[0]
        pt = q.get("prompt_type", "unknown")
        payload = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": _load_system_prompt(pt)},
                {"role": "user", "content": q.get("prompt_text", "")},
            ],
        }
        print("\n--- DRY RUN: first pending query payload ---")
        print(json.dumps(payload, indent=2))
        print(
            f"\n  pending: {len(pending_queries)}  already done: {len(completed_ids)}"
        )
        return

    # ---- Build OpenAI-compatible client (Groq or OpenRouter) ----
    provider = str(cfg.get("provider", "groq")).lower()
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            log.error("OPENROUTER_API_KEY not set — add it to .env or .env.local")
            sys.exit(1)
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/FitSenseAI",
                "X-Title": "FitSense AI Teacher Pipeline",
            },
        )
        log.info("  provider               : openrouter")
    else:
        # Default: Groq
        provider = "groq"
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            log.error("GROQ_API_KEY not set — add it to .env or .env.local")
            sys.exit(1)
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        log.info("  provider               : groq")

    # ---- Rate limiter (rolling 60-second input token window) ----
    input_tpm = int(cfg.get("input_tokens_per_minute", 6000))
    rate_limiter = InputTokenRateLimiter(tokens_per_minute=input_tpm)
    log.info("  input_tokens_per_minute: %d", input_tpm)

    # ---- Process ----
    all_records = list(existing_records)  # carry forward already-done records
    t_start = time.monotonic()

    n_success = sum(1 for r in existing_records if r["status"] == "success")
    n_failed = sum(1 for r in existing_records if r["status"] != "success")
    # Seed session token counters from existing records' raw_response usage
    sess_prompt = 0
    sess_completion = 0
    sess_total = 0

    pbar = tqdm(
        total=len(queries),
        initial=len(completed_ids),
        desc="Querying",
        unit="q",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    pbar.set_postfix(
        ok=n_success,
        fail=n_failed,
        in_tok=sess_prompt,
        out_tok=sess_completion,
        tot_tok=sess_total,
    )

    # Shared mutable state accessed from worker threads — protected by write_lock.
    write_lock = threading.Lock()

    def _handle_completed(
        record: dict, prompt_tok: int, completion_tok: int, total_tok: int
    ) -> None:
        """Called after each request completes. Must be called with write_lock held."""
        nonlocal sess_prompt, sess_completion, sess_total, n_success, n_failed

        sess_prompt += prompt_tok
        sess_completion += completion_tok
        sess_total += total_tok
        all_records.append(record)

        _append_jsonl(record, responses_path)

        is_failed = record["status"] != "success"
        if is_failed:
            _append_jsonl(record, failed_path)
            n_failed += 1
        else:
            n_success += 1
            # If this was a retry that succeeded, remove the old failed
            # entry from failed_responses.jsonl so it doesn't get re-queued
            # on the next run.
            _remove_query_from_failed(record["query_id"], failed_path)

        _rewrite_csv(all_records, csv_path)

        pbar.update(1)
        pbar.set_postfix(
            ok=n_success,
            fail=n_failed,
            in_tok=sess_prompt,
            out_tok=sess_completion,
            tot_tok=sess_total,
        )

    def _worker(query: dict) -> tuple[dict, int, int, int]:
        """Worker function: apply floor delay then call API."""
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        return process_query(
            query=query,
            client=client,
            model_name=model_name,
            provider=provider,
            source_run_id=source_run_id,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            rate_limiter=rate_limiter,
        )

    if max_workers == 1:
        # True sequential — simplest, no lock overhead
        for query in pending_queries:
            record, prompt_tok, completion_tok, total_tok = _worker(query)
            _handle_completed(record, prompt_tok, completion_tok, total_tok)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_query = {
                pool.submit(_worker, query): query for query in pending_queries
            }
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    record, prompt_tok, completion_tok, total_tok = future.result()
                except Exception as e:
                    # Unhandled exception in worker — create a failed record
                    record = {
                        "response_id": str(uuid.uuid4()),
                        "query_id": query.get("query_id", "unknown"),
                        "user_id": query.get("user_id", "unknown"),
                        "prompt_type": query.get("prompt_type", "unknown"),
                        "provider": provider,
                        "model_name": model_name,
                        "status": "failed",
                        "attempt_count": 0,
                        "request_payload": None,
                        "response_text": None,
                        "response_json": None,
                        "raw_response": {"error": traceback.format_exc()},
                        "source_query_run_id": source_run_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    prompt_tok = completion_tok = total_tok = 0

                with write_lock:
                    _handle_completed(record, prompt_tok, completion_tok, total_tok)

    pbar.close()

    elapsed = time.monotonic() - t_start

    # ---- Final summary ----
    summary = _write_summary(
        all_records,
        source_run_id,
        elapsed,
        cfg,
        summary_path,
        provider=provider,
        sess_prompt=sess_prompt,
        sess_completion=sess_completion,
        sess_total=sess_total,
    )

    tok = summary["tokens"]
    stok = summary["session_tokens"]

    log.info("")
    log.info("=== Run complete ===")
    log.info("  total        : %d", summary["total_queries"])
    log.info(
        "  success      : %d  (%.1f%%)",
        summary["success"],
        summary["success_rate"] * 100,
    )
    log.info("  json failed  : %d", summary["json_parse_failed"])
    log.info("  api failed   : %d", summary["api_failed"])
    log.info(
        "  elapsed      : %.1fs  (%.2f q/s)", elapsed, summary["queries_per_second"]
    )
    log.info("  --- session tokens (this run) ---")
    log.info("  input        : %d", stok["prompt_tokens"])
    log.info("  output       : %d", stok["completion_tokens"])
    log.info("  total        : %d", stok["total_tokens"])
    log.info("  --- cumulative tokens (all records) ---")
    log.info("  input        : %d", tok["prompt_tokens"])
    log.info("  output       : %d", tok["completion_tokens"])
    log.info("  total        : %d", tok["total_tokens"])
    log.info("  output dir   : %s", out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global log

    parser = argparse.ArgumentParser(
        description="Call Groq teacher LLM for FitSense queries (reads params.yaml)"
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap number of queries (smoke test)",
    )
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--request-delay-seconds", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _load_env()

    params_path = Path(args.params)
    if not params_path.exists():
        print(f"ERROR: params.yaml not found: {params_path}", file=sys.stderr)
        sys.exit(1)
    params = _load_params(params_path)

    # Set up logger now that we have params
    log = setup_logger(
        name="fitsense.teacher_llm",
        level=str(params["logging"]["level"]),
        log_dir=str(params["paths"]["logs_dir"]),
        file_name=str(params["logging"]["file_name"]),
        fmt=str(params["logging"]["format"]),
    )

    cli_overrides = {
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "timeout_seconds": args.timeout_seconds,
        "max_workers": args.max_workers,
        "max_queries": args.max_queries,
        "max_retries": args.max_retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "request_delay_seconds": args.request_delay_seconds,
    }
    cfg = _resolve_cfg(params, cli_overrides)
    paths_cfg = params.get("paths", {})

    run(cfg=cfg, paths_cfg=paths_cfg, params=params, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
