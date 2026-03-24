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
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTER_PATH = "Model-Pipeline/adapters/qwen-fitsense"
FORMATTED_BASE = Path("Model-Pipeline/data/formatted")
RUN_ID = "20260308T234052Z"
TEST_FILE = FORMATTED_BASE / RUN_ID / "test_formatted.jsonl"
REPORTS_DIR = Path("Model-Pipeline/reports")

# 3. Initialize W&B (Cloud persistence)
wandb.init(
    project="fitsense-model-pipeline",
    name=f"eval_student_{RUN_ID}",
    config={"model": "Qwen-2.5-7B-Student", "run_id": RUN_ID}
)

# 4. Load Model (Native Path for stability on T4)
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
    prompt = f"### Instruction:\nGenerate a FitSense workout plan.\n{user_message}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[1].strip() if "### Response:" in decoded else decoded.strip()

def main():
    with open(TEST_FILE, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    sample = random.sample(records, 10)
    preds, refs, json_valid_results = [], [], []

    for i, rec in enumerate(sample):
        log.info(f"Evaluating {i+1}/10...")
        prediction = call_model(rec["user_message"])
        preds.append(prediction)
        refs.append(rec["assistant"])
        
        # JSON Validity Check
        try:
            json.loads(prediction[prediction.find("{"):prediction.rfind("}")+1])
            json_valid_results.append(1)
        except:
            json_valid_results.append(0)

    # 5. Compute Advanced Metrics
    log.info("Computing ROUGE and BERTScore...")
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [r_scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)
    
    avg_metrics = {
        "json_validity_rate": sum(json_valid_results) / len(json_valid_results),
        "rougeL_mean": sum(rouge_l) / len(rouge_l),
        "bertscore_f1_mean": F1.mean().item()
    }

    # 6. Final Logging & Saving
    wandb.log(avg_metrics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / f"student_eval_{RUN_ID}.json"
    
    with open(report_file, "w") as f:
        json.dump({"run_id": RUN_ID, "metrics": avg_metrics}, f, indent=2)

    log.info(f"✅ Evaluation Complete. Report saved to {report_file}")
    wandb.finish()

if __name__ == "__main__":
    main()