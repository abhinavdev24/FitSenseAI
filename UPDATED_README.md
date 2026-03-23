# FitSense AI — Model Pipeline

This document describes the model development pipeline for FitSense AI, including what has been completed and what remains to be done.

---

## Overview

The model pipeline fine-tunes **Llama 3.1 8B Instruct** on the FitSense distillation dataset generated in Phase 1 (Data Pipeline). The student model learns to imitate the teacher model on fitness coaching tasks — specifically generating structured JSON workout plans.

---

## Directory Structure

```
Model-Pipeline/
  scripts/
    prepare_training_data.py     ← formats distillation dataset for Llama chat template
    evaluate_student.py          ← runs inference + computes ROUGE-L, BERTScore, JSON validity
    bias_detection.py            ← slice-based bias analysis using Fairlearn
    check_schema.py              ← quick schema validity check on eval report
  data/
    formatted/
      20260308T234052Z/
        train_formatted.jsonl    ← 632 training records (Llama chat template format)
        val_formatted.jsonl      ← 91 validation records
        test_formatted.jsonl     ← 77 test records
        manifest.json            ← metadata about formatting run
  reports/
    eval_report_20260308T234052Z.json    ← baseline evaluation results
    bias_report_20260308T234052Z.json    ← bias detection results (once run)
    bias_plots/                          ← bar charts per slice (once generated)
  adapters/
    final/                       ← fine-tuned LoRA adapter weights (download from Drive)
```

---

## What Has Been Completed ✅

### 1. Data Preparation (`prepare_training_data.py`)
- Loads distillation dataset from `Data-Pipeline/data/raw/distillation_dataset/20260308T234052Z/`
- Applies Llama 3.1 chat template to all records
- Preserves slice metadata: `prompt_type`, `goal_type`, `condition_flag`, `activity_level`
- Outputs formatted JSONL to `Model-Pipeline/data/formatted/20260308T234052Z/`

**To run:**
```bash
py Model-Pipeline/scripts/prepare_training_data.py
```

---

### 2. Distillation Dataset Builder (`Data-Pipeline/build_distillation_dataset.py`)
- Joins teacher responses with synthetic queries
- Extracts slice tags from prompt text
- Builds deterministic train/val/test splits (632/91/77)
- Outputs to `Data-Pipeline/data/raw/distillation_dataset/20260308T234052Z/`

**To run:**
```bash
py Data-Pipeline/build_distillation_dataset.py
```

---

### 3. Baseline Evaluation (`evaluate_student.py`)
- Runs inference on 77 test records using Llama 3.1 8B via Groq API
- Computes ROUGE-L, BERTScore, JSON validity, schema validity
- Logs metrics to W&B (needs valid W&B API key)
- Saves report to `Model-Pipeline/reports/eval_report_20260308T234052Z.json`

**Baseline results (zero-shot, before fine-tuning):**

| Metric | Score |
|---|---|
| ROUGE-L | 0.1657 |
| BERTScore F1 | 0.7718 |
| JSON Validity | 54.76% |
| Schema Validity | 35.06% |

**To run:**
```bash
$env:GROQ_API_KEY="your_groq_api_key"
py Model-Pipeline/scripts/evaluate_student.py
```

**Dependencies:**
```bash
pip install rouge-score bert-score wandb requests
```

---

### 4. Fine-Tuning (Google Colab)
- Fine-tuned Llama 3.1 8B Instruct using QLoRA (4-bit) on 632 training records
- Adapter weights saved to Google Drive at `MyDrive/fitsense-adapters/final/`

**Status:** Training complete. Adapter weights need to be downloaded from Google Drive and placed at:
```
Model-Pipeline/adapters/final/ -- placed
```

**Files in adapter folder:**
- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

---

## What Remains To Be Done ❌

### 1. Re-run Evaluation with Fine-Tuned Model (HIGH PRIORITY)

Update `evaluate_student.py` to load the local adapter instead of calling Groq API.

Replace the `call_model` function with local inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load once at module level
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "Model-Pipeline/adapters/final")
model.eval()

def call_model(system: str, user: str) -> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Re-run and compare before vs after fine-tuning scores.

---

### 2. Bias Detection (`bias_detection.py`) (HIGH PRIORITY)

Script is already written. Just needs to be run.

```bash
pip install fairlearn matplotlib
py Model-Pipeline/scripts/bias_detection.py
```

This will:
- Analyze 77 test records across 4 slices: `prompt_type`, `goal_type`, `condition_flag`, `activity_level`
- Generate 8 bar charts in `Model-Pipeline/reports/bias_plots/`
- Write `bias_report_20260308T234052Z.json`
- Flag any slices where performance drops significantly

---

### 3. Fix W&B Experiment Tracking (HIGH PRIORITY)

Current issue: W&B login is failing with 401 error.

Fix:
```bash
wandb login --relogin
# go to https://wandb.ai/authorize and paste API key
```

Then log the already-saved eval report:
```bash
py -c "
import json, wandb
with open('Model-Pipeline/reports/eval_report_20260308T234052Z.json') as f:
    report = json.load(f)
wandb.init(project='fitsense-model-pipeline', name='eval_20260308T234052Z')
wandb.log({
    'rougeL_mean': report['rouge']['rougeL_mean'],
    'bertscore_f1_mean': report['bertscore']['bertscore_f1_mean'],
    'json_validity_rate': report['json_validity']['json_validity_rate'],
    'schema_validity_rate': report['json_validity']['schema_validity_rate'],
})
wandb.finish()
"
```

---

### 4. Sensitivity Analysis (`sensitivity_analysis.py`) (MEDIUM PRIORITY)

Write a script that varies temperature (0.0, 0.2, 0.5, 0.8) and measures how JSON validity and ROUGE-L change. Reuses saved predictions — no additional API calls needed.

Output: bar charts showing metric sensitivity to temperature changes.

---

### 5. Push to GCP Artifact Registry (`push_to_registry.py`) (HIGH PRIORITY)

Push adapter weights + eval report to GCP Artifact Registry for versioning.

```python
from google.cloud import artifactregistry_v1
# or simply use GCS bucket:
# gsutil cp -r Model-Pipeline/adapters/final gs://fitsense-models/run_20260308T234052Z/
```

Requirements:
- GCP project: `mlops-gcp-lab-cloudrunner`
- Create a GCS bucket: `fitsense-models`
- Push only after eval + bias checks pass

---

### 6. Rollback Script (`rollback.py`) (MEDIUM PRIORITY)

Compare current eval report vs previous. Block registry push if new model performs worse.

```python
# Compare rougeL_mean and json_validity_rate
# If both are worse than previous → exit(1) to block CI/CD push
```

---

### 7. Dockerfile (HIGH PRIORITY)

Containerize the Model-Pipeline:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY Model-Pipeline/requirements.txt .
RUN pip install -r requirements.txt
COPY Model-Pipeline/ ./Model-Pipeline/
COPY Data-Pipeline/data/ ./Data-Pipeline/data/
CMD ["py", "Model-Pipeline/scripts/evaluate_student.py"]
```

---

### 8. CI/CD GitHub Actions (HIGH PRIORITY)

Create `.github/workflows/model-pipeline-ci.yml`:

```yaml
name: Model Pipeline CI
on:
  push:
    paths:
      - 'Model-Pipeline/**'
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r Model-Pipeline/requirements.txt
      - name: Run evaluation
        run: py Model-Pipeline/scripts/evaluate_student.py
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  bias_check:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - name: Run bias detection
        run: py Model-Pipeline/scripts/bias_detection.py
  push_registry:
    needs: bias_check
    runs-on: ubuntu-latest
    steps:
      - name: Push to GCP
        run: py Model-Pipeline/scripts/push_to_registry.py
```

---

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install rouge-score bert-score wandb requests fairlearn matplotlib transformers peft bitsandbytes accelerate
```

---

## Key Run IDs

| Item | Run ID |
|---|---|
| Distillation dataset | `20260308T234052Z` |
| Synthetic profiles | `20260308T234033Z` |
| Eval report | `20260308T234052Z` |

---

## Environment Variables Required

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API for inference |
| `WANDB_API_KEY` | W&B experiment tracking |
| `DISTILLATION_RUN_ID` | Override auto-detected run ID |
| `GCS_BUCKET` | GCS bucket for model registry push |

---

## Assignment Requirements Checklist

| Requirement | Status |
|---|---|
| Load data from data pipeline | ✅ Done |
| Model validation with metrics | ✅ Done (baseline) |
| Fine-tuned model | ✅ Done (1 epoch, adapters in Drive) |
| Re-evaluate fine-tuned model | ❌ Remaining |
| Bias detection with slicing | ❌ Remaining |
| Experiment tracking (W&B) | ⚠️ Partial (auth issue) |
| Sensitivity analysis | ❌ Remaining |
| Push to GCP Artifact Registry | ❌ Remaining |
| Rollback mechanism | ❌ Remaining |
| Dockerfile | ❌ Remaining |
| CI/CD GitHub Actions | ❌ Remaining |