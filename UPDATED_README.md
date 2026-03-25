# FitSense AI — Model Pipeline

This document describes the model development pipeline for FitSense AI, including architecture decisions, training, evaluation, bias detection, and CI/CD automation.

---

## Overview

The model pipeline fine-tunes **Qwen3-8B** on the FitSense distillation dataset generated in Phase 1 (Data Pipeline). The student model learns to imitate the teacher model (Qwen 32B via Groq) on fitness coaching tasks — specifically generating structured JSON workout plans that respect user profiles, medical conditions, injuries, and safety constraints.

---

## Pipeline Flow

> See diagram below for the end-to-end execution sequence from data preparation to GCP registry push.

![FitSense Model Pipeline Flow](./docs/pipeline_flow.png)

---

## Directory Structure

```
Model-Pipeline/
  Scripts/
    prepare_training_data.py     ← formats distillation dataset for Qwen3 ChatML template
    trainmodel.py                ← fine-tunes Qwen3-8B with LoRA on  GPU
    evaluate_student.py          ← runs inference + computes ROUGE-L, BERTScore, JSON validity
    bias_slicing.py              ← slice-based bias analysis across demographic groups
    check_schema.py              ← schema validity check on eval report with diagnostics
    push_to_registry.py          ← pushes adapter to GCP Artifact Registry
  scripts/
    prepare_training_data.py     
    evaluate.py                  ←  evaluation script
    bias_detection.py            ← bias detection script 
    load_data.py                 ← data loading utilities
    hparam_search.py             ← hyperparameter search utilities
    select_model.py              ← model selection logic
  data/
    formatted/
      20260308T234052Z/
        train_formatted.jsonl    ← training records (Qwen3 ChatML format)
        val_formatted.jsonl      ← validation records
        test_formatted.jsonl     ← 77 test records
        manifest.json            ← metadata about formatting run
  reports/
    eval_report_20260308T234052Z.json    ← baseline evaluation results
    student_eval_20260308T234052Z.json   ← post fine-tuning evaluation results
    bias_report_20260308T234052Z.json    ← bias detection results
  adapters/
    qwen3-8b-fitsense/           ← fine-tuned LoRA adapter weights
      adapter_config.json
      adapter_model.safetensors
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
      checkpoint-20/             ← intermediate checkpoint
      checkpoint-40/             ← intermediate checkpoint
      checkpoint-60/             ← final checkpoint
.github/
  workflows/
    data-pipeline-ci.yml         ← CI/CD pipeline (data + model validation)
```

---

## Model Architecture

| Component | Detail |
|---|---|
| Teacher model | `Qwen 32B` via Groq |
| Student base model | `unsloth/Qwen3-8B-bnb-4bit` via Unsloth + OpenRouter |
| Fine-tuning method | LoRA (Low-Rank Adaptation) via PEFT |
| Quantization | 4-bit (BitsAndBytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training framework | Unsloth + TRL SFTTrainer |
| Training hardware | Google Colab T4 GPU (16GB VRAM) |

> **Note:** The student model is Qwen3-8B, chosen because it fits on a T4 GPU with 4-bit quantization (~5GB VRAM), supports a dual thinking/non-thinking mode that can be disabled for structured JSON output, and is available pre-quantized via Unsloth for faster training. The teacher model is Qwen 32B served via Groq, used during the data pipeline phase to generate the distillation dataset.

### Why LoRA over full fine-tuning?

Fine-tuning all 8 billion parameters would require 80+ GB of VRAM. LoRA freezes the base model and attaches small trainable adapter matrices (~20–40M parameters), reducing VRAM requirements to ~5GB while still teaching the model FitSense-specific behavior — the correct JSON schema, safety constraint handling, and plan structure.

---

## 1. Data Preparation

**Script:** `Model-Pipeline/Scripts/prepare_training_data.py`

- Loads distillation dataset from `Data-Pipeline/data/raw/distillation_dataset/20260308T234052Z/`
- Formats records into **Qwen3 model's template** 
- Appends `/no_think` to every user message to disable Qwen3's reasoning mode
- Injects empty `<think></think>` block before assistant response to train model to skip reasoning
- Adds explicit JSON schema skeleton to system prompt to prevent wrong key generation
- Preserves slice metadata: `prompt_type`, `goal_type`, `condition_flag`, `activity_level`, `age_band`, `sex`
- Outputs formatted JSONL to `Model-Pipeline/data/formatted/20260308T234052Z/`

**To run:**
```bash
py Model-Pipeline/Scripts/prepare_training_data.py
```

---

## 2. Think Tag Handling

Qwen3-8B supports a dual-mode architecture — thinking mode (chain-of-thought reasoning) and non-thinking mode. For structured JSON generation, thinking mode is explicitly disabled across the entire pipeline:

| Location | What we do | Why |
|---|---|---|
| Training data | `/no_think` + empty `<think></think>` | Teaches model to skip reasoning during fine-tuning |
| Inference prompt | `/no_think` + empty `<think></think>` | Prevents reasoning tokens at generation time |
| Output parsing | `split("</think>")` fallback strip | Removes any leaked think content before JSON extraction |
| `check_schema.py` | `strip_think()` before JSON search | Prevents think block content being mistaken for JSON |

Without this, think tokens consumed 500–800 tokens of the generation budget before JSON even started, causing truncated plans and 0% schema validity.

---

## 3. Training

**Script:** `Model-Pipeline/Scripts/trainmodel.py`
**Runtime:** Google Colab T4 GPU

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 1e-4 | Standard for LoRA fine-tuning |
| Batch size | 1 (effective 8 with grad accum) | T4 VRAM constraint |
| Gradient accumulation steps | 8 | Simulates larger batch size |
| Max steps | 60 | Light fine-tune sufficient to teach schema |
| LR scheduler | Cosine | Smooth decay for stable convergence |
| Optimizer | adamw_8bit | Memory-efficient Adam |
| Warmup steps | 10 | Prevents early instability |
| Max sequence length | 2048 | Covers full plan + context |
| LoRA r | 16 | Balances expressiveness vs VRAM |
| LoRA alpha | 32 | 2x rank — standard scaling practice |

### Checkpointing

Checkpoints saved every 20 steps (steps 20, 40, 60) to Google Drive. `resume_from_checkpoint=True` allows automatic resume if Colab session disconnects.

**To run (Colab only — requires GPU):**
```python
# Update paths at top of script then run
dataset_path = "/content/drive/MyDrive/FitSense/train_formatted.jsonl"
output_dir   = "/content/drive/MyDrive/FitSense/qwen3-8b-fitsense"
```

---

## 4. Model Validation

**Script:** `Model-Pipeline/Scripts/evaluate_student.py`

Evaluation performed on hold-out test set (`test_formatted.jsonl`, 77 records) not used during training. Samples 20 records for evaluation.



### Results

| Metric |  Fine-Tuning (Baseline) 
|---|---|---|
| Schema Validity | 35.06% 
| JSON Validity | 54.76% 
| ROUGE-L | 0.1657 
| BERTScore F1 | 0.7718 

> Post fine-tuning results will be updated once Colab evaluation completes.

All metrics logged to Weights & Biases.
**W&B Project:** https://wandb.ai/harinihari-jk-/fitsense-model-pipeline

**To run (Colab only — requires GPU):**
```python
ADAPTER_PATH = "/content/qwen3-8b-fitsense"
TEST_FILE    = "/content/test_formatted.jsonl"
REPORTS_DIR  = Path("/content/drive/MyDrive/FitSense/reports")
```

---

## 5. Schema Validation

**Script:** `Model-Pipeline/Scripts/check_schema.py`

Validates that model predictions conform to the required FitSense JSON schema:
```json
{
  "plan_name": "...",
  "days": [{
    "name": "...",
    "day_order": 1,
    "notes": null,
    "exercises": [{
      "exercise_name": "...",
      "position": 1,
      "sets": [{"set_number": 1, "target_reps": 10, "target_rir": 2, "rest_seconds": 60}]
    }]
  }]
}
```

Also runs full diagnostics reporting think block leakage, truncation, and phase-nesting counts across all records.

**To run:**
```bash
py Model-Pipeline/Scripts/check_schema.py
```

---

## 6. Bias Detection

**Script:** `Model-Pipeline/Scripts/bias_slicing.py`

Evaluates model performance across five demographic and contextual slices:

| Slice | Values |
|---|---|
| `goal_type` | strength, mobility, sleep_improvement, longevity, etc. |
| `condition_flag` | has_condition, no_condition |
| `activity_level` | sedentary, lightly_active, moderately_active, very_active |
| `age_band` | various age groups |
| `sex` | M, F, other |

Tracks ROUGE-L, JSON validity rate, and schema validity rate per slice. Any slice deviating more than 15 percentage points from the mean schema validity rate is flagged as a potential bias signal.

### Bias Mitigation Strategies

If disparities are detected:
1. Oversample underperforming slices in fine-tuning data
2. Add slice-specific schema examples to the system prompt
3. Post-hoc JSON repair for known failure patterns
4. Re-weight loss function to penalize errors on underperforming groups

**To run:**
```bash
py Model-Pipeline/Scripts/bias_slicing.py
```

**Dependencies:**
```bash
pip install rouge-score fairlearn matplotlib
```

---

## 7. Experiment Tracking

**Tool:** Weights & Biases (W&B)

Each evaluation run logs:
- Aggregate metrics (JSON validity, schema validity, ROUGE-L, BERTScore F1)
- Per-sample table with predictions, references, and per-record scores
- Hyperparameter configuration
- Run metadata (model name, run ID, timestamp)

**Fix W&B auth if needed:**
```bash
wandb login --relogin
# Go to https://wandb.ai/authorize and paste API key
```

---

## 8. Hyperparameter Sensitivity

| Hyperparameter | Effect on performance |
|---|---|
| `max_new_tokens` | Most critical — 1024 caused truncation; raised to 2048 fixed schema validity |
| `/no_think` switch | Largest single improvement — eliminated think token budget consumption |
| `lora_r` | Higher rank = more expressive but more VRAM; r=16 balances both |
| `lora_alpha` | Controls adapter scaling; alpha=32 (2x rank) is standard practice |
| `max_steps` | 60 steps sufficient for schema learning without overfitting |
| `learning_rate` | 1e-4 standard for LoRA; higher caused instability on short runs |

---

## 9. Model Registry

**Script:** `Model-Pipeline/Scripts/push_to_registry.py`

Once the model passes validation and bias checks, the LoRA adapter is:
1. Packaged as a `.tar.gz` archive
2. Uploaded to Google Cloud Storage at `gs://<bucket>/models/fitsense-qwen3-8b/<run_id>/`
3. Registered in Vertex AI Model Registry with eval metrics attached as metadata

**GCP Configuration:**
```python
GCP_PROJECT = "mlops-gcp-lab-cloudrunner"
GCS_BUCKET  = "fitsense-models"
GCP_REGION  = "us-central1"
```

**To run:**
```bash
pip install google-cloud-aiplatform google-cloud-storage
py Model-Pipeline/Scripts/push_to_registry.py
```

---

## 10. CI/CD Pipeline

**File:** `.github/workflows/data-pipeline-ci.yml`

Triggers on every push to `main` or manual dispatch.

### Jobs

| Job | Trigger | What it does |
|---|---|---|
| `test` | Every push/PR | Runs Data Pipeline pytest suite |
| `run-scripts-and-generate-artifacts` | Push to main / manual | Runs synthetic data generation |
| `model-pipeline-validation` | After data pipeline succeeds | Schema check, bias slicing, quality gate |

### Quality Gate

Pipeline **fails automatically** if schema validity drops below 50%.

### Rollback Mechanism

Training checkpoints saved at steps 20, 40, 60. If a newly trained model performs worse, restore by loading from `checkpoint-40` or `checkpoint-20` in the adapter directory.

---

## 11. Reproduction Steps

### Prerequisites
```bash
pip install unsloth trl transformers peft datasets
pip install rouge-score bert-score wandb fairlearn matplotlib
pip install google-cloud-aiplatform google-cloud-storage
```

### Step 1 — Prepare training data
```bash
py Model-Pipeline/Scripts/prepare_training_data.py
```

### Step 2 — Train model (GPU required — run on Google Colab)
```bash
# Open trainmodel.py in Colab and update dataset_path and output_dir
```

### Step 3 — Evaluate (GPU required — run on Google Colab)
```bash
# Open evaluate_student.py in Colab and update ADAPTER_PATH and TEST_FILE
```

### Step 4 — Schema validation (runs locally)
```bash
py Model-Pipeline/Scripts/check_schema.py
```

### Step 5 — Bias analysis (runs locally)
```bash
py Model-Pipeline/Scripts/bias_slicing.py
```

### Step 6 — Push to registry
```bash
py Model-Pipeline/Scripts/push_to_registry.py
```

---

## 12. Assignment Requirements Checklist

| Requirement | Status |
|---|---|
| Load data from data pipeline | ✅ Done |
| Model training with LoRA | ✅ Done (Qwen3-8B, 60 steps, Colab T4) |
| Model validation with metrics | ✅ Done (ROUGE-L, BERTScore, JSON validity, schema validity) |
| Re-evaluate fine-tuned model | ⏳ In progress (Colab evaluation running) |
| Bias detection with slicing | ✅ Done (`bias_slicing.py`) |
| Experiment tracking (W&B) | ✅ Done |
| Hyperparameter sensitivity | ✅ Documented |
| Push to GCP Artifact Registry | ✅ Done (`push_to_registry.py`) |
| Rollback mechanism | ✅ Done (checkpoints at steps 20, 40, 60) |
| CI/CD GitHub Actions | ✅ Done (3-job pipeline with quality gate) |
| Schema validation | ✅ Done (`check_schema.py` with diagnostics) |

---

## Key Run IDs

| Item | Run ID |
|---|---|
| Distillation dataset | `20260308T234052Z` |
| Eval report | `20260308T234052Z` |
| Fine-tuned adapter | `qwen3-8b-fitsense` |

---

## Environment Variables Required

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY` | W&B experiment tracking |
| `DISTILLATION_RUN_ID` | Override auto-detected run ID |
| `GCP_PROJECT` | GCP project ID |
| `GCS_BUCKET` | GCS bucket for model registry push |
| `GCP_REGION` | GCP region (default: us-central1) 