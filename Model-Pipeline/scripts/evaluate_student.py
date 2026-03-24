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
BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"
ADAPTER_PATH = "Model-Pipeline/adapters/qwen3-8b-fitsense"
FORMATTED_BASE = Path("Model-Pipeline/data/formatted")
RUN_ID = "20260308T234052Z"
TEST_FILE = FORMATTED_BASE / RUN_ID / "test_formatted.jsonl"
REPORTS_DIR = Path("Model-Pipeline/reports")

# 3. Initialize W&B
wandb.init(
    project="fitsense-model-pipeline",
    name=f"eval_student_{RUN_ID}",
    config={"model": "Qwen3-8B-Student", "run_id": RUN_ID}
)

# 4. Load Model
log.info("Loading model and adapter...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()


def call_model(user_message):
    """
    Generate using ChatML template with /no_think to disable Qwen3 thinking mode.
    The empty <think></think> block forces the model to skip reasoning and output
    JSON directly, preventing think tokens from consuming the token budget.
    As a fallback, any leaked </think> content is stripped before returning.
    """
    prompt = (
        f"<|im_start|>user\n{user_message} /no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n</think>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,   # raised from 1024 — plans need room to breathe
            do_sample=False
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Strip any residual think block (fallback safety)
    if "</think>" in decoded:
        decoded = decoded.split("</think>", 1)[-1].strip()

    return decoded.strip()


def main():
    with open(TEST_FILE, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    sample = random.sample(records, min(10, len(records)))
    preds, refs, json_valid_results = [], [], []

    for i, rec in enumerate(sample):
        log.info(f"Evaluating {i+1}/{len(sample)}...")
        prediction = call_model(rec["user_message"])
        preds.append(prediction)
        refs.append(rec["assistant"])

        # JSON Validity Check
        try:
            start = prediction.find("{")
            end = prediction.rfind("}")
            if start != -1 and end != -1:
                json.loads(prediction[start:end + 1])
                json_valid_results.append(1)
            else:
                json_valid_results.append(0)
        except Exception:
            json_valid_results.append(0)

    # 5. Compute Metrics
    log.info("Computing ROUGE and BERTScore...")
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [r_scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]

    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)

    avg_metrics = {
        "json_validity_rate": sum(json_valid_results) / len(json_valid_results),
        "rougeL_mean": sum(rouge_l) / len(rouge_l),
        "bertscore_f1_mean": F1.mean().item()
    }

    # 6. Log & Save
    wandb.log(avg_metrics)

    table = wandb.Table(columns=["sample_id", "user_message", "prediction", "reference", "rougeL", "bertscore_f1", "json_valid"])
    for i in range(len(sample)):
        table.add_data(
            i + 1,
            sample[i]["user_message"][:300],
            preds[i][:500],
            refs[i][:500],
            round(rouge_l[i], 4),
            round(F1[i].item(), 4),
            json_valid_results[i],
        )
    wandb.log({"eval_samples": table})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / f"student_eval_{RUN_ID}.json"

    with open(report_file, "w") as f:
        json.dump({"run_id": RUN_ID, "metrics": avg_metrics}, f, indent=2)

    log.info(f"✅ Evaluation Complete. Report saved to {report_file}")
    wandb.finish()


if __name__ == "__main__":
    main()