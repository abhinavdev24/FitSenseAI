"""
build_distillation_dataset.py
------------------------------
Joins teacher responses with queries to build the distillation dataset
for student model training/evaluation.

Input:
  Data-Pipeline/data/raw/synthetic_queries/<run_id>/queries.jsonl
  Data-Pipeline/data/raw/teacher-llm-responses/<run_id>/responses.jsonl

Output:
  Data-Pipeline/data/raw/distillation_dataset/<run_id>/
    all_records.jsonl
    train.jsonl
    val.jsonl
    test.jsonl
    manifest.json
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

RAW_BASE         = Path("Data-Pipeline/data/raw")
QUERIES_BASE     = RAW_BASE / "synthetic_queries"
RESPONSES_BASE   = RAW_BASE / "teacher-llm-responses"
OUTPUT_BASE      = RAW_BASE / "distillation_dataset"

RUN_ID           = "20260308T234052Z"

# Train/val/test split ratios
TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.10
# TEST_RATIO     = 0.10 (remainder)

SEED             = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── Slice tag extraction from prompt_text ─────────────────────────────────────

def extract_slice_tags(prompt_text: str) -> dict:
    """
    Parse slice tags from the embedded prompt_text profile block.
    Falls back to 'unknown' if a field is not found.
    """
    def find(pattern, text, default="unknown"):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    age_raw      = find(r"Age:\s*(\d+)", prompt_text)
    sex_raw      = find(r"Sex:\s*(\w+)", prompt_text)
    activity_raw = find(r"Activity level:\s*([^\n]+)", prompt_text)
    goals_raw    = find(r"Goals \(priority order\):\s*([^\n]+)", prompt_text)
    conditions   = find(r"Medical conditions:\s*([^\n]+)", prompt_text)

    # Age band
    try:
        age = int(age_raw)
        if age < 25:
            age_band = "18-24"
        elif age < 35:
            age_band = "25-34"
        elif age < 45:
            age_band = "35-44"
        elif age < 55:
            age_band = "45-54"
        else:
            age_band = "55+"
    except ValueError:
        age_band = "unknown"

    # Sex normalisation
    sex = sex_raw.upper() if sex_raw != "unknown" else "unknown"

    # Activity level — take first word
    activity_level = activity_raw.split()[0].lower() if activity_raw != "unknown" else "unknown"

    # Goal type — take first goal listed
    goal_type = goals_raw.split(",")[0].strip().lower() if goals_raw != "unknown" else "unknown"

    # Condition flag
    condition_flag = "no_condition" if conditions.lower() in ("none", "unknown", "") else "has_condition"

    return {
        "age_band":       age_band,
        "sex":            sex,
        "activity_level": activity_level,
        "goal_type":      goal_type,
        "condition_flag": condition_flag,
    }


# ── Split assignment (deterministic, hash-based) ──────────────────────────────

def assign_split(record_id: str) -> str:
    """Deterministic split assignment based on record_id hash."""
    h = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 100
    if h < TRAIN_RATIO * 100:
        return "train"
    elif h < (TRAIN_RATIO + VAL_RATIO) * 100:
        return "val"
    else:
        return "test"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_queries(run_id: str) -> dict:
    """Load queries keyed by query_id."""
    path = QUERIES_BASE / run_id / "queries.jsonl"
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                queries[q["query_id"]] = q
    log.info(f"Loaded {len(queries)} queries from {path}")
    return queries


def load_responses(run_id: str) -> list:
    """Load successful responses only."""
    path = RESPONSES_BASE / run_id / "responses.jsonl"
    responses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("status") == "success" and r.get("response_json"):
                    responses.append(r)
    log.info(f"Loaded {len(responses)} successful responses from {path}")
    return responses


# ── Builder ───────────────────────────────────────────────────────────────────

def build_record(response: dict, query: dict) -> dict:
    """Combine response + query into a distillation record."""
    prompt_text = query.get("prompt_text", "")
    slice_tags  = extract_slice_tags(prompt_text)

    # Use response_json (already parsed) as the label
    response_str = json.dumps(response["response_json"])

    record_id = response["response_id"]

    return {
        "record_id":   record_id,
        "query_id":    response["query_id"],
        "user_id":     response.get("user_id", ""),
        "instruction": prompt_text,
        "context": {
            "prompt_type":                 response["prompt_type"],
            "slice_tags":                  slice_tags,
            "expected_safety_constraints": [],   # can be enriched later
            "context_summary":             f"User prompt type: {response['prompt_type']}. "
                                           f"Goal: {slice_tags['goal_type']}. "
                                           f"Activity: {slice_tags['activity_level']}. "
                                           f"Condition: {slice_tags['condition_flag']}.",
        },
        "response": response_str,
        "metadata": {
            "provider":      response.get("provider", ""),
            "model_name":    response.get("model_name", ""),
            "attempt_count": response.get("attempt_count", 1),
            "source_run_id": RUN_ID,
            "created_at":    response.get("created_at", ""),
        },
        "split": assign_split(record_id),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Building distillation dataset for run_id: {RUN_ID}")

    queries   = load_queries(RUN_ID)
    responses = load_responses(RUN_ID)

    # Join responses with queries
    records   = {"train": [], "val": [], "test": []}
    skipped   = 0

    for resp in responses:
        qid   = resp.get("query_id")
        query = queries.get(qid)
        if not query:
            log.warning(f"No query found for query_id={qid}, skipping.")
            skipped += 1
            continue

        record = build_record(resp, query)
        records[record["split"]].append(record)

    total = sum(len(v) for v in records.values())
    log.info(f"Built {total} records | train={len(records['train'])} "
             f"val={len(records['val'])} test={len(records['test'])} skipped={skipped}")

    # Write outputs
    out_dir = OUTPUT_BASE / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for split, recs in records.items():
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        all_records.extend(recs)
        log.info(f"Wrote {len(recs)} records to {path}")

    # Write all_records.jsonl
    with open(out_dir / "all_records.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # Manifest
    manifest = {
        "run_id":        RUN_ID,
        "created_at":    datetime.utcnow().isoformat() + "Z",
        "total_records": total,
        "splits": {
            "train": len(records["train"]),
            "val":   len(records["val"]),
            "test":  len(records["test"]),
        },
        "skipped":       skipped,
        "prompt_types":  list({r["context"]["prompt_type"] for r in all_records}),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Manifest written to {out_dir / 'manifest.json'}")
    log.info("Distillation dataset build complete.")


if __name__ == "__main__":
    main()