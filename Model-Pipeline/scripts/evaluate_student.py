import json
import os
import torch
import logging
import random
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import wandb

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 2. Configuration
BASE_MODEL   = "unsloth/Qwen3-4B-bnb-4bit"
ADAPTER_PATH = "Model-Pipeline/adapters/qwen3-4b-fitsense"
TEST_FILE    = "Model-Pipeline/data/formatted/test.jsonl"
REPORTS_DIR  = Path("Model-Pipeline/reports")
RUN_ID       = "20260331Z"

# 3. Initialize W&B
wandb.init(
    project="fitsense-model-pipeline",
    name=f"eval_student_4b_{RUN_ID}",
    config={"model": "Qwen3-4B-Student", "run_id": RUN_ID}
)

# 4. Load Model
log.info("Loading Qwen3-4B model and adapter...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()


def call_model(user_message):
    """Generate using ChatML template with /no_think."""
    prompt = (
        f"<|im_start|>user\n{user_message} /no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n</think>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if "</think>" in decoded:
        decoded = decoded.split("</think>", 1)[-1].strip()
    return decoded.strip()


def extract_user_message(rec):
    """Extract user message from request_payload.messages."""
    try:
        messages = rec["request_payload"]["messages"]
        # Get the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg["content"]
    except (KeyError, TypeError):
        pass
    return ""


def extract_assistant_response(rec):
    """Extract assistant response — prefer response_json, fallback to response_text."""
    # Try response_json first (already parsed JSON object)
    resp_json = rec.get("response_json")
    if resp_json:
        if isinstance(resp_json, str):
            return resp_json.strip()
        if isinstance(resp_json, dict):
            return json.dumps(resp_json)

    # Fallback to response_text
    resp_text = rec.get("response_text", "")
    if resp_text:
        # Strip think tags if present
        if "<think>" in resp_text:
            resp_text = resp_text.split("</think>")[-1].strip()
        return resp_text.strip()

    return ""


def main():
    # Load test file
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Could not find test file at {TEST_FILE}")

    with open(TEST_FILE, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Filter only successful records
    records = [r for r in records if r.get("status") == "success"]
    log.info(f"Found {len(records)} successful records.")

    sample = random.sample(records, min(20, len(records)))
    preds, refs, json_valid_results = [], [], []

    for i, rec in enumerate(sample):
        log.info(f"Evaluating {i+1}/{len(sample)}...")

        user_message   = extract_user_message(rec)
        assistant_ref  = extract_assistant_response(rec)

        if not user_message or not assistant_ref:
            log.warning(f"Record {i} missing content. Skipping.")
            continue

        prediction = call_model(user_message)
        preds.append(prediction)
        refs.append(assistant_ref)

        # JSON Validity Check
        try:
            start, end = prediction.find("{"), prediction.rfind("}")
            if start != -1 and end != -1:
                json.loads(prediction[start:end + 1])
                json_valid_results.append(1)
            else:
                json_valid_results.append(0)
        except Exception:
            json_valid_results.append(0)

    if not preds:
        log.error("❌ ERROR: No samples were successfully parsed. Check your dataset keys.")
        return

    # 5. Compute Metrics
    log.info("Computing Metrics...")
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l  = [r_scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)

    avg_metrics = {
        "json_validity_rate": sum(json_valid_results) / len(json_valid_results),
        "rougeL_mean":        sum(rouge_l) / len(rouge_l),
        "bertscore_f1_mean":  F1.mean().item()
    }

    log.info(f"Metrics: {avg_metrics}")

    # 6. Log & Save
    wandb.log(avg_metrics)
    table = wandb.Table(columns=["sample_id", "prediction", "reference", "rougeL", "bertscore_f1", "json_valid"])
    for i in range(len(preds)):
        table.add_data(i+1, preds[i][:500], refs[i][:500], round(rouge_l[i], 4), round(F1[i].item(), 4), json_valid_results[i])
    wandb.log({"eval_samples": table})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    per_record = []
    for i in range(len(preds)):
        per_record.append({
            "record_id":    i + 1,
            "prompt_type":  sample[i].get("prompt_type", "plan_creation"),
            "prediction":   preds[i],
            "reference":    refs[i],
            "rougeL":       round(rouge_l[i], 4),
            "bertscore_f1": round(F1[i].item(), 4),
            "json_valid":   bool(json_valid_results[i]),
        })
    report_file = REPORTS_DIR / f"student_eval_{RUN_ID}.json"
    with open(report_file, "w") as f:
        json.dump({"run_id": RUN_ID, "metrics": avg_metrics, "per_record": per_record}, f, indent=2)

    log.info(f"✅ Evaluation Complete. Report saved to {report_file}")
    wandb.finish()


if __name__ == "__main__":
    main()