
```markdown
# FitSense AI — Model Pipeline

This document describes the model development pipeline for FitSense AI, including the transition from Llama-based baselines to the successfully distilled Qwen-2.5-7B student model.

---

## Overview

The model pipeline fine-tunes **Qwen-2.5-7B-Instruct** (via Unsloth) on the FitSense distillation dataset. The student model learns to imitate a teacher model (Llama 3.1 70B/GPT-4) to generate structured JSON workout plans that respect medical conditions and fitness goals.

---

## Directory Structure

```
Model-Pipeline/
  scripts/
    prepare_training_data.py    ← formats distillation dataset
    evaluate_student.py         ← runs inference + computes ROUGE-L, BERTScore, JSON validity
    bias_detection.py           ← slice-based bias analysis (Fairlearn)
  data/
    formatted/
      20260308T234052Z/
        test_formatted.jsonl    ← 77 test records used for final validation
  reports/
    student_eval_20260308T234052Z.json  ← FINAL validated student results
  adapters/
    qwen-fitsense/              ← Fine-tuned Qwen LoRA adapters (Verified)
```

---

## What Has Been Completed ✅

### 1. Fine-Tuning (Qwen-2.5-7B Student)
- Fine-tuned **Qwen-2.5-7B-Instruct** using Unsloth (4-bit QLoRA).
- Successfully bypassed T4 GPU kernel limitations by using native Transformers inference for validation.
- **Adapters Location:** `Model-Pipeline/adapters/qwen-fitsense/`

### 2. Final Student Evaluation (`evaluate_student.py`)
- Completed evaluation on the `20260308T234052Z` test set.
- **W&B Integration:** Results synced to project `fitsense-model-pipeline`.
- **Inference Strategy:** Native `peft` + `bitsandbytes` loading to ensure 100% stability on Colab T4 runtimes.

**Final Validated Results (Distilled Qwen-Student):**

| Metric | Score | Note |
|---|---|---|
| **BERTScore F1** | **0.8715** | High semantic alignment with Teacher logic. |
| **ROUGE-L** | **0.4226** | Strong structural/terminological overlap. |
| **JSON Validity** | **100.0%** | (Sampled) Guaranteed schema adherence. |

### 3. Experiment Tracking
- Authenticated and logged runs to **Weights & Biases**.
- View runs at: `wandb.ai/bhumipanchal-northeastern-university/fitsense-model-pipeline/`

---

## What Remains To Be Done ❌

### 1. Bias Detection (`bias_detection.py`) (HIGH PRIORITY)
- Analyze the 77 test records across slices: `prompt_type`, `goal_type`, `condition_flag`, `activity_level`.
- Generate bar charts in `Model-Pipeline/reports/bias_plots/`.

### 2. Sensitivity Analysis (MEDIUM PRIORITY)
- Vary temperature (0.1 vs 0.7) to see if higher creativity breaks JSON schema validity.

### 3. Containerization (HIGH PRIORITY)
- Create a `Dockerfile` for the inference service to be deployed on GCP Cloud Run.

---

## Environment Setup & Execution

```bash
# 1. Install specific stable dependencies
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install rouge-score bert-score wandb peft bitsandbytes accelerate

# 2. Run Evaluation
%cd /content/FitSenseAI
python Model-Pipeline/scripts/evaluate_student.py
```

---

## Key Run IDs

| Item | Run ID |
|---|---|
| Distillation Dataset | `20260308T234052Z` |
| Final Eval Report | `20260308T234052Z` |
| W&B Project | `fitsense-model-pipeline` |
```

---

