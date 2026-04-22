#!/usr/bin/env python3
"""
FitSenseAI — Deployment Monitor
================================
Polls the deployed vLLM endpoint with validation samples, computes inference
metrics, detects model decay, and optionally detects data drift.

Metrics computed:
  - json_validity_rate    : fraction of responses that are valid JSON
  - schema_compliance     : fraction with required keys (plan_name + days)
  - avg_latency_ms        : average wall-clock time per request (ms)
  - thinking_presence_rate: fraction of responses that include <think> block

Thresholds (default):
  - json_validity_rate  >= 0.70  (triggers retraining if below)
  - avg_latency_ms      <= 8000  (alert only — no retraining)

Exit codes:
  0 — all thresholds passed
  1 — one or more thresholds failed (CI treats this as "retrain needed")

Usage examples:
  # Run against live GCE endpoint
  python monitor_deployment.py \\
    --endpoint http://<gce-external-ip>:8000 \\
    --api-key  $VLLM_API_KEY

  # Dry run (mock responses, no HTTP calls)
  python monitor_deployment.py --endpoint http://localhost:8000 --api-key x --dry-run

  # Compare against training baseline
  python monitor_deployment.py \\
    --endpoint http://<gce-external-ip>:8000 --api-key $VLLM_API_KEY \\
    --baseline Model-Pipeline/outputs/evaluation/evaluation_results.json
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional HTTP client — prefer httpx, fall back to stdlib urllib
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx  # type: ignore[import]
    _USE_HTTPX = True
except ImportError:
    import urllib.request as _urllib_req  # type: ignore[import]
    _USE_HTTPX = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAL_PATH = REPO_ROOT / "Model-Pipeline" / "data" / "training" / "val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Model-Deployment" / "reports"
DRIFT_LOG = REPO_ROOT / "Model-Deployment" / "reports" / "request_log.jsonl"

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "json_validity_rate":    {"min": 0.70, "trigger_retrain": True},
    "schema_compliance":     {"min": 0.60, "trigger_retrain": True},
    "avg_latency_ms":        {"max": 8000, "trigger_retrain": False},  # alert only
}

# ---------------------------------------------------------------------------
# Built-in test prompts (used when val.jsonl is not available)
# ---------------------------------------------------------------------------
BUILT_IN_PROMPTS = [
    {
        "system": (
            "You are FitSense AI, an expert fitness coach. "
            "Return only a JSON object with plan_name and days array. "
            "No markdown fences, no explanation."
        ),
        "user": (
            "Generate a workout plan for me.\n\n"
            "## My Profile\n"
            "Age: 28, Sex: M\n"
            "Activity level: moderately_active\n"
            "Goals (priority order): muscle_gain\n"
            "Medical conditions: none\nInjuries: none\nMedications: none\n\n"
            "## Instructions\n"
            "- Do NOT include any weights.\n"
            "- Return a valid JSON object and nothing else."
        ),
    },
    {
        "system": (
            "You are FitSense AI, an expert fitness coach. "
            "Return only a JSON object with plan_name and days array."
        ),
        "user": (
            "Generate a workout plan for me.\n\n"
            "## My Profile\n"
            "Age: 45, Sex: F\n"
            "Activity level: lightly_active\n"
            "Goals (priority order): weight_loss\n"
            "Medical conditions: hypertension\nInjuries: none\nMedications: Lisinopril 10mg\n\n"
            "## Instructions\n"
            "- Avoid heavy compound lifts due to hypertension.\n"
            "- Return a valid JSON object and nothing else."
        ),
    },
    {
        "system": (
            "You are FitSense AI. Return only JSON with plan_name and days."
        ),
        "user": (
            "Generate a workout plan.\n\n"
            "## My Profile\n"
            "Age: 22, Sex: other\n"
            "Activity level: very_active\n"
            "Goals: endurance\n"
            "Medical conditions: none\nInjuries: knee_sprain\nMedications: none\n\n"
            "Return a valid JSON object only."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_think(text: str) -> str:
    """Remove <think>...</think> block from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def is_valid_json(text: str) -> bool:
    cleaned = strip_think(text)
    try:
        json.loads(cleaned)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def has_schema(text: str) -> bool:
    """Check that the JSON has plan_name and days keys."""
    cleaned = strip_think(text)
    try:
        obj = json.loads(cleaned)
        return isinstance(obj, dict) and "plan_name" in obj and "days" in obj
    except (json.JSONDecodeError, ValueError):
        return False


def has_thinking(text: str) -> bool:
    return bool(re.search(r"<think>.*?</think>", text, flags=re.DOTALL))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def call_endpoint(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int = 90,
) -> tuple[str, float]:
    """
    Send a chat completion request.
    Returns (response_text, latency_ms).
    Raises RuntimeError on HTTP / connection error.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    try:
        if _USE_HTTPX:
            with _httpx.Client(timeout=timeout) as client:  # type: ignore[union-attr]
                resp = client.post(url, content=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        else:
            req = _urllib_req.Request(  # type: ignore[attr-defined]
                url, data=payload, headers=headers, method="POST"
            )
            with _urllib_req.urlopen(req, timeout=timeout) as resp:  # type: ignore[attr-defined]
                data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"HTTP call failed: {exc}") from exc

    latency_ms = (time.monotonic() - t0) * 1000
    content = data["choices"][0]["message"]["content"]
    return content, latency_ms


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def load_prompts(val_path: Path, n_samples: int) -> list[dict[str, str]]:
    """Load up to n_samples prompts from val.jsonl. Falls back to built-ins."""
    if not val_path.exists():
        print(f"  [WARN] val.jsonl not found at {val_path}, using built-in prompts")
        return BUILT_IN_PROMPTS[:n_samples]

    prompts: list[dict[str, str]] = []
    with open(val_path, encoding="utf-8") as fh:
        for raw_line in fh:
            if len(prompts) >= n_samples:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                system_content = next(
                    (m["content"] for m in msgs if m.get("role") == "system"), ""
                )
                user_content = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                if system_content and user_content:
                    prompts.append({"system": system_content, "user": user_content})
            except (json.JSONDecodeError, KeyError):
                continue

    if not prompts:
        print("  [WARN] No valid prompts loaded from val.jsonl, using built-ins")
        return BUILT_IN_PROMPTS[:n_samples]

    print(f"  Loaded {len(prompts)} prompts from {val_path}")
    return prompts


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def detect_drift(
    current_prompts: list[dict[str, str]],
    val_path: Path,
) -> dict[str, Any]:
    """
    Stub: compares prompt length distribution of current batch vs val set.
    A full implementation would compare prompt_type tags from metadata.
    """
    if not val_path.exists():
        return {"status": "skipped", "reason": "val.jsonl not found"}

    # Compare average user message length as a simple proxy for distribution shift
    def avg_len(prompts: list[dict[str, str]]) -> float:
        lengths = [len(p.get("user", "")) for p in prompts]
        return sum(lengths) / len(lengths) if lengths else 0.0

    current_avg = avg_len(current_prompts)

    val_prompts: list[dict[str, str]] = []
    with open(val_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                user_content = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                if user_content:
                    val_prompts.append({"user": user_content})
            except (json.JSONDecodeError, KeyError):
                continue

    val_avg = avg_len(val_prompts)

    if val_avg == 0:
        return {"status": "skipped", "reason": "empty val set"}

    relative_shift = abs(current_avg - val_avg) / val_avg
    flagged = relative_shift > 0.25

    return {
        "status": "flagged" if flagged else "ok",
        "current_avg_prompt_len": round(current_avg, 1),
        "val_avg_prompt_len": round(val_avg, 1),
        "relative_shift": round(relative_shift, 4),
        "threshold": 0.25,
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Main monitoring logic
# ---------------------------------------------------------------------------

def run_monitor(
    endpoint: str,
    api_key: str,
    model: str,
    prompts: list[dict[str, str]],
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Run inference on all prompts and return (aggregate_metrics, per_sample_results).
    """
    results: list[dict[str, Any]] = []

    for i, prompt in enumerate(prompts, start=1):
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user",   "content": prompt["user"]},
        ]

        if dry_run:
            # Simulate a valid response
            mock_content = (
                '<think>\nMock reasoning trace.\n</think>\n'
                '{"plan_name": "Mock Plan", "days": [{"name": "DAY_1", "day_order": 1, '
                '"notes": "", "exercises": []}]}'
            )
            result: dict[str, Any] = {
                "sample_idx": i,
                "latency_ms": 500.0,
                "response": mock_content,
                "json_valid": True,
                "schema_compliant": True,
                "has_thinking": True,
                "error": None,
            }
        else:
            try:
                response_text, latency_ms = call_endpoint(
                    endpoint, api_key, model, messages
                )
                result = {
                    "sample_idx": i,
                    "latency_ms": round(latency_ms, 1),
                    "response": response_text[:500],  # truncate for report
                    "json_valid": is_valid_json(response_text),
                    "schema_compliant": has_schema(response_text),
                    "has_thinking": has_thinking(response_text),
                    "error": None,
                }
                print(
                    f"  [{i}/{len(prompts)}] latency={latency_ms:.0f}ms  "
                    f"json={result['json_valid']}  "
                    f"schema={result['schema_compliant']}  "
                    f"think={result['has_thinking']}"
                )
            except RuntimeError as exc:
                result = {
                    "sample_idx": i,
                    "latency_ms": None,
                    "response": None,
                    "json_valid": False,
                    "schema_compliant": False,
                    "has_thinking": False,
                    "error": str(exc),
                }
                print(f"  [{i}/{len(prompts)}] ERROR: {exc}")

        results.append(result)

    # Aggregate
    valid_latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    n = len(results)
    metrics: dict[str, Any] = {
        "n_samples": n,
        "json_validity_rate":     round(sum(r["json_valid"] for r in results) / n, 4) if n else 0.0,
        "schema_compliance":      round(sum(r["schema_compliant"] for r in results) / n, 4) if n else 0.0,
        "thinking_presence_rate": round(sum(r["has_thinking"] for r in results) / n, 4) if n else 0.0,
        "avg_latency_ms":         round(sum(valid_latencies) / len(valid_latencies), 1) if valid_latencies else None,
        "error_rate":             round(sum(1 for r in results if r["error"]) / n, 4) if n else 0.0,
    }

    return metrics, results


def check_thresholds(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """
    Return (all_passed, list_of_violation_messages).
    If baseline is provided, also check for >= 10% regression vs baseline.
    """
    violations: list[str] = []

    for metric_name, rules in thresholds.items():
        value = metrics.get(metric_name)
        if value is None:
            continue
        if "min" in rules and value < rules["min"]:
            violations.append(
                f"{metric_name}={value:.3f} < threshold {rules['min']:.3f}"
                + (" [RETRAIN]" if rules.get("trigger_retrain") else " [ALERT]")
            )
        if "max" in rules and value > rules["max"]:
            violations.append(
                f"{metric_name}={value:.1f} > threshold {rules['max']}"
                + (" [RETRAIN]" if rules.get("trigger_retrain") else " [ALERT]")
            )

    # Regression check vs baseline
    if baseline:
        baseline_metrics = baseline.get("metrics", {})
        for key in ("json_validity_rate", "schema_compliance"):
            current = metrics.get(key)
            base_val = baseline_metrics.get(key)
            if current is not None and base_val is not None and base_val > 0:
                regression = (base_val - current) / base_val
                if regression > 0.10:
                    violations.append(
                        f"{key} regressed {regression:.1%} vs baseline "
                        f"({base_val:.3f} → {current:.3f}) [RETRAIN]"
                    )

    retrain_violations = [v for v in violations if "[RETRAIN]" in v]
    return len(retrain_violations) == 0, violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FitSenseAI deployment monitor — detects model decay and data drift"
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="vLLM base URL, e.g. http://<gce-external-ip>:8000"
    )
    parser.add_argument("--api-key", required=True, help="vLLM API key")
    parser.add_argument("--model", default="fitsense", help="Model alias (default: fitsense)")
    parser.add_argument(
        "--val-path", type=Path, default=DEFAULT_VAL_PATH,
        help="Path to val.jsonl for test prompts"
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Path to evaluation_results.json for regression comparison"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of val samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write monitoring reports"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip actual HTTP calls; generate mock passing report"
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 60)
    print("FitSenseAI — Deployment Monitor")
    print("=" * 60)
    print(f"  Endpoint  : {args.endpoint}")
    print(f"  Model     : {args.model}")
    print(f"  N samples : {args.n_samples}")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  Timestamp : {ts}")
    print()

    # Load prompts
    prompts = load_prompts(args.val_path, args.n_samples)

    # Load baseline if provided
    baseline: dict[str, Any] | None = None
    if args.baseline and args.baseline.exists():
        with open(args.baseline, encoding="utf-8") as fh:
            baseline = json.load(fh)
        print(f"  Baseline  : {args.baseline}")
    else:
        print("  Baseline  : none (absolute thresholds only)")
    print()

    # Run inference
    print("Running inference samples...")
    metrics, per_sample = run_monitor(
        args.endpoint, args.api_key, args.model, prompts, args.dry_run
    )

    # Drift detection
    print("\nRunning drift detection...")
    drift_result = detect_drift(prompts, args.val_path)
    print(f"  Drift status: {drift_result.get('status', 'unknown')}")
    if drift_result.get("flagged"):
        print(f"  WARNING: prompt length distribution shifted "
              f"{drift_result['relative_shift']:.1%} vs training baseline")

    # Check thresholds
    thresholds_passed, violations = check_thresholds(metrics, DEFAULT_THRESHOLDS, baseline)

    # Print metric summary
    print("\n" + "─" * 40)
    print("Metrics summary:")
    print(f"  json_validity_rate    : {metrics['json_validity_rate']:.1%}  "
          f"(threshold ≥ {DEFAULT_THRESHOLDS['json_validity_rate']['min']:.0%})")
    print(f"  schema_compliance     : {metrics['schema_compliance']:.1%}  "
          f"(threshold ≥ {DEFAULT_THRESHOLDS['schema_compliance']['min']:.0%})")
    print(f"  thinking_presence_rate: {metrics['thinking_presence_rate']:.1%}")
    if metrics["avg_latency_ms"] is not None:
        print(f"  avg_latency_ms        : {metrics['avg_latency_ms']:.0f}ms  "
              f"(threshold ≤ {DEFAULT_THRESHOLDS['avg_latency_ms']['max']}ms)")
    print(f"  error_rate            : {metrics['error_rate']:.1%}")

    print()
    if violations:
        print("Violations:")
        for v in violations:
            print(f"  ✗ {v}")
    else:
        print("  All thresholds passed ✓")

    overall = "PASSED" if thresholds_passed else "FAILED"
    print(f"\nResult: {overall}")

    # Write report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"monitor_{ts}.json"
    report: dict[str, Any] = {
        "timestamp": ts,
        "endpoint": args.endpoint,
        "model": args.model,
        "dry_run": args.dry_run,
        "thresholds_passed": thresholds_passed,
        "metrics": metrics,
        "violations": violations,
        "drift": drift_result,
        "thresholds_used": DEFAULT_THRESHOLDS,
        "per_sample_results": per_sample,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report written to: {report_path}")

    return 0 if thresholds_passed else 1


if __name__ == "__main__":
    sys.exit(main())
