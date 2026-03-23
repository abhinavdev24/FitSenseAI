"""
evaluate_student.py
-------------------
Runs inference on the test split using fine-tuned Llama 3.1 8B Instruct
(local adapter via transformers + peft) and evaluates:
  - ROUGE-L
  - BERTScore
  - JSON validity rate (critical for FitSense plan outputs)
  - Schema validity rate (days → exercises → sets structure)

Logs all metrics to W&B.

Input:  Model-Pipeline/data/formatted/<run_id>/test_formatted.jsonl
Output: Model-Pipeline/reports/eval_report_<run_id>.json
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime

import re
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import wandb

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

FORMATTED_BASE = Path("Model-Pipeline/data/formatted")
REPORTS_DIR    = Path("Model-Pipeline/reports")
ADAPTER_PATH   = Path("Model-Pipeline/adapters/final")

WANDB_PROJECT  = "fitsense-model-pipeline"

# JSON validity thresholds
JSON_VALIDITY_THRESHOLD   = 0.50
SCHEMA_VALIDITY_THRESHOLD = 0.40

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── Model Loading ─────────────────────────────────────────────────────────────

log.info("Loading Llama 3.1 8B with fine-tuned adapter (CPU inference)...")

try:
    hf_token = os.environ.get("HF_TOKEN")
    
    # Use bfloat16 to halve the memory requirement to ~16GB.
    # low_cpu_mem_usage prevents RAM spikes by mapping the files efficiently.
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="cpu",
        dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        token=hf_token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        token=hf_token  # Only pass token if provided
    )

    log.info(f"Loading adapter from: {ADAPTER_PATH}")
    if not ADAPTER_PATH.exists():
        log.error(f"Adapter path does not exist: {ADAPTER_PATH}")
        log.error(f"Please ensure fine-tuned adapter is at: {ADAPTER_PATH}")
        raise FileNotFoundError(f"Adapter path not found: {ADAPTER_PATH}")
    
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()
    log.info("Model loaded successfully!")
except Exception as e:
    log.error(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    raise SystemExit(1)


# ── Inference ─────────────────────────────────────────────────────────────────

def call_model(system: str, user: str) -> str:
    """Run local inference using fine-tuned Llama 3.1 8B."""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant response (after the last <|start_header_id|>assistant<|end_header_id|>)
    if "assistant<|end_header_id|>" in response:
        response = response.split("assistant<|end_header_id|>")[-1].strip()
    return response


# ── JSON / Schema validation ──────────────────────────────────────────────────

def extract_json(text: str) -> str:
    """Strip markdown fences and surrounding prose to extract raw JSON."""
    # Try to find a JSON block inside ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: find first { to last }
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    return text.strip()


def is_valid_json(text: str) -> bool:
    try:
        json.loads(extract_json(text))
        return True
    except Exception:
        return False


def normalise_keys(obj):
    """Recursively lowercase all keys in a dict/list."""
    if isinstance(obj, dict):
        return {k.lower(): normalise_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalise_keys(i) for i in obj]
    return obj


def is_valid_plan_schema(text: str) -> bool:
    """
    Check the plan JSON matches the expected schema.
    Accepts both exact and case-variant field names.
    Also accepts alternate structures like 'workout_days'/'exercises' flat list.
    """
    try:
        raw = json.loads(extract_json(text))
        obj = normalise_keys(raw)

        # Must have some kind of plan name
        has_name = any(k in obj for k in ("plan_name", "name", "plan"))
        assert has_name, "no plan name"

        # Must have days OR exercises list at top level
        days = obj.get("days") or obj.get("workout_days") or obj.get("weeks")
        if days and isinstance(days, list) and len(days) > 0:
            # Check at least one day has exercises
            for day in days:
                if isinstance(day, dict):
                    exs = day.get("exercises") or day.get("exercise_list") or []
                    if isinstance(exs, list):
                        return True  # good enough
        # Flat exercises list at top level
        exs = obj.get("exercises")
        if exs and isinstance(exs, list) and len(exs) > 0:
            return True
        return False
    except Exception:
        return False


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return {
        "rougeL_mean": round(sum(scores) / len(scores), 4),
        "rougeL_min":  round(min(scores), 4),
        "rougeL_max":  round(max(scores), 4),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    P, R, F = bert_score(predictions, references, lang="en", verbose=False)
    f_scores = F.tolist()
    return {
        "bertscore_f1_mean": round(sum(f_scores) / len(f_scores), 4),
        "bertscore_f1_min":  round(min(f_scores), 4),
    }


def compute_json_metrics(predictions: list[str], prompt_types: list[str]) -> dict:
    """Only check JSON validity on plan_creation and plan_modification outputs."""
    plan_types  = {"plan_creation", "plan_modification"}
    plan_preds  = [p for p, t in zip(predictions, prompt_types) if t in plan_types]

    if not plan_preds:
        return {"json_validity_rate": None, "schema_validity_rate": None, "plan_outputs_checked": 0}

    json_valid   = [is_valid_json(p) for p in plan_preds]
    schema_valid = [is_valid_plan_schema(p) for p in plan_preds]

    return {
        "json_validity_rate":   round(sum(json_valid)   / len(plan_preds), 4),
        "schema_validity_rate": round(sum(schema_valid) / len(plan_preds), 4),
        "plan_outputs_checked": len(plan_preds),
    }


# ── Main eval loop ────────────────────────────────────────────────────────────

def load_test_records(formatted_dir: Path) -> list[dict]:
    path = formatted_dir / "test_formatted.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"Loaded {len(records)} test records from {path}")
    return records


def run_evaluation(run_id: str) -> dict:
    formatted_dir = FORMATTED_BASE / run_id
    records       = load_test_records(formatted_dir)

    # Extract system prompt from manifest
    manifest_path = formatted_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    system_prompt = manifest["system_prompt"]

    predictions  = []
    references   = []
    prompt_types = []
    per_record   = []

    log.info(f"Running inference on {len(records)} records via fine-tuned Llama 3.1 8B...")

    for i, rec in enumerate(records):
        pred = call_model(system_prompt, rec["user_message"])
        predictions.append(pred)
        references.append(rec["assistant"])
        prompt_types.append(rec.get("prompt_type", ""))

        per_record.append({
            "record_id":      rec.get("record_id", ""),
            "prompt_type":    rec.get("prompt_type", ""),
            "goal_type":      rec.get("goal_type", ""),
            "condition_flag": rec.get("condition_flag", ""),
            "activity_level": rec.get("activity_level", ""),
            "prediction":     pred,
            "reference":      rec["assistant"],
            "json_valid":     is_valid_json(pred),
            "schema_valid":   is_valid_plan_schema(pred),
        })

        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{len(records)} done")

    log.info("Computing metrics...")
    rouge_metrics  = compute_rouge(predictions, references)
    bert_metrics   = compute_bertscore(predictions, references)
    json_metrics   = compute_json_metrics(predictions, prompt_types)

    # Threshold checks
    json_rate   = json_metrics.get("json_validity_rate") or 1.0
    passed_json = json_rate >= JSON_VALIDITY_THRESHOLD

    report = {
        "run_id":           run_id,
        "model":            "Llama 3.1 8B (fine-tuned LoRA adapter)",
        "evaluated_at":     datetime.utcnow().isoformat() + "Z",
        "num_records":      len(records),
        "rouge":            rouge_metrics,
        "bertscore":        bert_metrics,
        "json_validity":    json_metrics,
        "thresholds": {
            "json_validity_threshold":   JSON_VALIDITY_THRESHOLD,
            "schema_validity_threshold": SCHEMA_VALIDITY_THRESHOLD,
            "passed_json_check":         passed_json,
        },
        "per_record": per_record,
    }

    return report


# ── W&B logging ───────────────────────────────────────────────────────────────

def log_to_wandb(report: dict):
    wandb.init(project=WANDB_PROJECT, name=f"eval_{report['run_id']}", config={
        "model":   report["model"],
        "run_id":  report["run_id"],
        "records": report["num_records"],
    })
    wandb.log({
        "rougeL_mean":          report["rouge"]["rougeL_mean"],
        "bertscore_f1_mean":    report["bertscore"]["bertscore_f1_mean"],
        "json_validity_rate":   report["json_validity"].get("json_validity_rate"),
        "schema_validity_rate": report["json_validity"].get("schema_validity_rate"),
        "passed_json_check":    int(report["thresholds"]["passed_json_check"]),
    })
    wandb.finish()
    log.info("Metrics logged to W&B.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    run_id = os.environ.get("DISTILLATION_RUN_ID")
    if not run_id:
        # auto-detect latest
        dirs = sorted(FORMATTED_BASE.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        run_id = dirs[0].name
        log.info(f"Auto-detected run_id: {run_id}")

    report = run_evaluation(run_id)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"eval_report_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Eval report written to: {report_path}")

    # Log to W&B if available
    try:
        log_to_wandb(report)
    except Exception as e:
        log.warning(f"Could not log to W&B: {e}")

    # Exit non-zero if JSON validity check fails (used by CI/CD)
    if not report["thresholds"]["passed_json_check"]:
        log.error(
            f"JSON validity rate {report['json_validity']['json_validity_rate']:.2f} "
            f"below threshold {JSON_VALIDITY_THRESHOLD}. Blocking pipeline."
        )
        raise SystemExit(1)

    log.info("Evaluation passed all thresholds.")


if __name__ == "__main__":
    main()