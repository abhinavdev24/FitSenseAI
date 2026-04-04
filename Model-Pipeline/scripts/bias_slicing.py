"""
bias_slicing.py
---------------
Evaluates model performance across demographic slices to detect bias.
Slices by: goal_type, condition_flag, activity_level, age_band, sex.

Input:  Model-Pipeline/reports/student_eval_<run_id>.json  (per-sample results)
        Model-Pipeline/data/formatted/<run_id>/test_formatted.jsonl
Output: Model-Pipeline/reports/bias_report_<run_id>.json
        Console table of per-slice metrics
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from rouge_score import rouge_scorer

# ── Config ────────────────────────────────────────────────────────────────────
RUN_ID        = "20260401Z"
FORMATTED_DIR = Path("Model-Pipeline/data/formatted/20260308T234052Z")
EVAL_REPORT   = Path("Model-Pipeline/reports/student_eval_20260331Z.json")
OUTPUT_DIR    = Path("Model-Pipeline/reports")

SLICE_KEYS = ["goal_type", "condition_flag", "activity_level", "age_band", "sex"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_think(t: str) -> str:
    if "</think>" in t:
        t = t.split("</think>", 1)[-1].strip()
    return t

def extract_json(t: str) -> str:
    t = strip_think(t)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        return m.group(1)
    s, e = t.find("{"), t.rfind("}")
    return t[s:e+1] if s != -1 and e > s else t

def is_json_valid(text: str) -> int:
    try:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1:
            json.loads(text[s:e+1])
            return 1
    except Exception:
        pass
    return 0

def is_schema_valid(text: str) -> int:
    try:
        from collections.abc import Mapping
        obj = json.loads(extract_json(text))

        def nk(o):
            if isinstance(o, dict):
                return {k.lower(): nk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [nk(i) for i in o]
            return o

        obj = nk(obj)
        if "workout_plan" in obj and isinstance(obj["workout_plan"], dict):
            obj = obj["workout_plan"]
        elif "plan" in obj and isinstance(obj["plan"], dict):
            obj = obj["plan"]

        days = (obj.get("days") or obj.get("workout_days") or
                obj.get("training_days") or obj.get("schedule") or [])
        if not isinstance(days, list) or len(days) == 0:
            return 0
        for d in days:
            if not isinstance(d, dict):
                continue
            exs = (d.get("exercises") or d.get("exercise_list") or
                   d.get("workout") or d.get("movements") or [])
            if isinstance(exs, list) and len(exs) > 0:
                return 1
        return 0
    except Exception:
        return 0

# ── Load data ─────────────────────────────────────────────────────────────────

def load_eval_report():
    with open(EVAL_REPORT) as f:
        return json.load(f)

def load_formatted(split="test"):
    path = FORMATTED_DIR / f"{split}_formatted.jsonl"
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["record_id"]] = rec
    return records

# ── Compute per-record metrics ────────────────────────────────────────────────

def compute_metrics(report, formatted):
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    for rec in report["per_record"]:
        rid        = rec["record_id"]
        prediction = rec["prediction"]
        reference  = rec.get("reference", "")

        # Pull slice metadata from formatted data if available, else from report
        meta = formatted.get(rid, {})
        slices = {k: (meta.get(k) or rec.get(k) or "unknown") for k in SLICE_KEYS}

        rouge_l      = r_scorer.score(reference, prediction)["rougeL"].fmeasure
        json_valid   = is_json_valid(prediction)
        schema_valid = is_schema_valid(prediction)

        results.append({
            **slices,
            "record_id":    rid,
            "prompt_type":  rec.get("prompt_type", ""),
            "rougeL":       round(rouge_l, 4),
            "json_valid":   json_valid,
            "schema_valid": schema_valid,
        })

    return results

# ── Slice aggregation ─────────────────────────────────────────────────────────

def aggregate_by_slice(results, slice_key):
    buckets = defaultdict(list)
    for r in results:
        buckets[r[slice_key]].append(r)

    summary = {}
    for val, recs in sorted(buckets.items()):
        n = len(recs)
        summary[val] = {
            "n":                  n,
            "rougeL_mean":        round(sum(r["rougeL"] for r in recs) / n, 4),
            "json_validity_rate": round(sum(r["json_valid"] for r in recs) / n, 4),
            "schema_validity_rate": round(sum(r["schema_valid"] for r in recs) / n, 4),
        }
    return summary

# ── Disparity detection ───────────────────────────────────────────────────────

def detect_disparities(slice_summary, threshold=0.15):
    """Flag any slice whose schema_validity_rate deviates more than
    `threshold` from the overall mean — potential bias signal."""
    rates = [v["schema_validity_rate"] for v in slice_summary.values()]
    if not rates:
        return []
    mean_rate = sum(rates) / len(rates)
    flagged = []
    for slice_val, metrics in slice_summary.items():
        deviation = abs(metrics["schema_validity_rate"] - mean_rate)
        if deviation > threshold:
            flagged.append({
                "slice_value": slice_val,
                "schema_validity_rate": metrics["schema_validity_rate"],
                "mean_rate": round(mean_rate, 4),
                "deviation": round(deviation, 4),
            })
    return flagged

# ── Pretty print ──────────────────────────────────────────────────────────────

def print_slice_table(slice_key, summary):
    print(f"\n{'='*60}")
    print(f"  Slice: {slice_key}")
    print(f"{'='*60}")
    print(f"  {'Value':<30} {'N':>4}  {'ROUGE-L':>8}  {'JSON%':>6}  {'Schema%':>8}")
    print(f"  {'-'*58}")
    for val, m in summary.items():
        print(
            f"  {str(val):<30} {m['n']:>4}  "
            f"{m['rougeL_mean']:>8.4f}  "
            f"{m['json_validity_rate']:>6.2%}  "
            f"{m['schema_validity_rate']:>8.2%}"
        )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading eval report: {EVAL_REPORT}")
    report    = load_eval_report()
    formatted = load_formatted("test")

    results = compute_metrics(report, formatted)
    print(f"\nTotal records evaluated: {len(results)}")

    full_report = {
        "run_id":  RUN_ID,
        "n_total": len(results),
        "slices":  {},
        "disparities": {},
        "mitigation_notes": (
            "Disparities above 15pp deviation from mean schema validity rate "
            "are flagged. Suggested mitigations: (1) oversample underperforming "
            "slices in fine-tuning data, (2) add slice-specific schema examples "
            "to the system prompt, (3) apply post-hoc JSON repair for known "
            "failure patterns."
        )
    }

    for key in SLICE_KEYS:
        summary     = aggregate_by_slice(results, key)
        disparities = detect_disparities(summary)

        print_slice_table(key, summary)

        if disparities:
            print(f"\n  ⚠️  Disparities detected in '{key}':")
            for d in disparities:
                print(f"     {d['slice_value']}: {d['schema_validity_rate']:.2%} "
                      f"(mean={d['mean_rate']:.2%}, Δ={d['deviation']:.2%})")
        else:
            print(f"\n  ✅ No significant disparities in '{key}' (threshold=15pp)")

        full_report["slices"][key]      = summary
        full_report["disparities"][key] = disparities

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"bias_report_{RUN_ID}.json"
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n✅ Bias report saved to {out_path}")

if __name__ == "__main__":
    main()