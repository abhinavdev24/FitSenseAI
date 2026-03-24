# FitSense AI — Model Pipeline

## Overview

FitSense AI is a personalized fitness coaching system built on a teacher-student distillation architecture. A large teacher model (Llama 3.1 70B via OpenRouter) generates 50,000 synthetic training examples which are used to fine-tune a smaller student model (Qwen3-8B) for efficient, production-ready inference. The student model produces structured JSON workout plans that respect user profiles, medical conditions, injuries, and safety constraints.

---

## Project Structure

```
FitSenseAI/
├── Data-Pipeline/
│   ├── generate_synthetic_profiles.py
│   ├── generate_synthetic_workouts.py
│   ├── generate_synthetic_queries.py
│   ├── build_distillation_dataset.py
│   ├── validate.py
│   └── data/
│       └── raw/
│           ├── synthetic_profiles/
│           ├── synthetic_workouts/
│           ├── synthetic_queries/
│           └── distillation_dataset/
│               └── 20260308T234052Z/
│                   ├── train.jsonl
│                   ├── val.jsonl
│                   └── test.jsonl
├── Model-Pipeline/
│   ├── Scripts/
│   │   ├── prepare_training_data.py
│   │   ├── trainmodel.py
│   │   ├── evaluate_student.py
│   │   ├── check_schema.py
│   │   ├── bias_slicing.py
│   │   └── push_to_registry.py
│   ├── data/
│   │   └── formatted/
│   │       └── 20260308T234052Z/
│   │           ├── train_formatted.jsonl
│   │           ├── val_formatted.jsonl
│   │           ├── test_formatted.jsonl
│   │           └── manifest.json
│   ├── adapters/
│   │   └── qwen3-8b-fitsense/
│   └── reports/
│       ├── eval_report_20260308T234052Z.json
│       ├── student_eval_20260308T234052Z.json
│       └── bias_report_20260308T234052Z.json
└── .github/
    └── workflows/
        └── data-pipeline-ci.yml
```

---

## 1. Data Pipeline Integration

Training data is loaded directly from the output of the Data Pipeline — a versioned distillation dataset identified by `run_id: 20260308T234052Z`. The `prepare_training_data.py` script reads from:

```
Data-Pipeline/data/raw/distillation_dataset/<run_id>/{train,val,test}.jsonl
```

Each record is formatted into Qwen3 ChatML format with:
- A structured system prompt containing an explicit JSON schema skeleton
- `/no_think` appended to every user turn to disable Qwen3's reasoning mode
- An empty `<think></think>` block before the assistant response to train the model to skip reasoning and output JSON directly

A `manifest.json` is written alongside the formatted splits recording the system prompt, token statistics, and run metadata for reproducibility.

---

## 2. Model Architecture

| Component | Detail |
|---|---|
| Base model | `unsloth/Qwen3-8B-bnb-4bit` |
| Fine-tuning method | LoRA (Low-Rank Adaptation) via PEFT |
| Quantization | 4-bit (BitsAndBytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training framework | Unsloth + TRL SFTTrainer |
| Training hardware | Google Colab T4 GPU |

### Why LoRA?

Fine-tuning all 8 billion parameters would require 80+ GB of VRAM. LoRA freezes the base model and attaches small trainable adapter matrices (~20-40M parameters), reducing VRAM requirements to ~5GB while still teaching the model FitSense-specific behavior — the correct JSON schema, safety constraint handling, and plan structure.

---

## 3. Training

**Script:** `Model-Pipeline/Scripts/trainmodel.py`

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 1e-4 | Standard for LoRA fine-tuning |
| Batch size | 1 (effective 8 with grad accum) | T4 VRAM constraint |
| Gradient accumulation steps | 8 | Simulates larger batch |
| Max steps | 60 | Light fine-tune sufficient to teach schema |
| LR scheduler | Cosine | Smooth decay for stable convergence |
| Optimizer | adamw_8bit | Memory-efficient Adam |
| Warmup steps | 10 | Prevents early instability |
| Max sequence length | 2048 | Covers full plan + context |

### Checkpointing

Checkpoints are saved every 20 steps (steps 20, 40, 60) to prevent data loss from Colab session disconnections. `resume_from_checkpoint=True` allows training to automatically resume from the latest checkpoint if interrupted.

### Think Tag Handling

Qwen3-8B supports a dual-mode architecture — thinking mode (chain-of-thought reasoning) and non-thinking mode. For structured JSON generation, thinking mode is explicitly disabled:

- `/no_think` is appended to every user message as a soft switch
- An empty `<think></think>` block is injected in training data so the model learns to skip reasoning entirely
- At inference time, a fallback `split("</think>")` strips any leaked think content before JSON extraction

This prevents think tokens from consuming the `max_new_tokens` budget and causing truncated JSON output.

---

## 4. Model Validation

**Script:** `Model-Pipeline/Scripts/evaluate_student.py`

Evaluation is performed on a hold-out test set (`test_formatted.jsonl`) not used during training. Metrics computed per record and aggregated:

| Metric | Description |
|---|---|
| JSON Validity Rate | % of predictions containing parseable JSON |
| Schema Validity Rate | % of predictions matching the required plan schema |
| ROUGE-L | Longest common subsequence overlap with reference |
| BERTScore F1 | Semantic similarity using contextual embeddings |

### Results

| Metric | Pre Fine-Tuning (Baseline) | Post Fine-Tuning |
|---|---|---|
| Schema Validity | 35.06% | TBD after eval |
| JSON Validity | 54.76% | TBD after eval |
| ROUGE-L | 0.1657 | TBD after eval |
| BERTScore F1 | 0.7718 | TBD after eval |

All metrics are logged to Weights & Biases for experiment tracking and visualization.

---

## 5. Experiment Tracking

**Tool:** Weights & Biases (W&B)

**Project:** `fitsense-model-pipeline`

Each evaluation run logs:
- Aggregate metrics (JSON validity, schema validity, ROUGE-L, BERTScore F1)
- Per-sample table with predictions, references, and per-record scores
- Hyperparameter configuration
- Run metadata (model name, run ID, timestamp)

W&B dashboard: https://wandb.ai/harinihari-jk-/fitsense-model-pipeline

---

## 6. Hyperparameter Sensitivity

The key hyperparameters and their sensitivity:

| Hyperparameter | Effect |
|---|---|
| `max_new_tokens` | Directly controls truncation — too low (1024) causes JSON cut-off; set to 2048 |
| `lora_r` (rank) | Higher rank = more expressive adapter but more VRAM; r=16 balances both |
| `lora_alpha` | Controls adapter scaling; alpha=32 (2x rank) is standard practice |
| `max_steps` | 60 steps is a light fine-tune — sufficient for schema learning, not overfitting |
| `learning_rate` | 1e-4 chosen as standard LoRA rate; too high causes instability on short runs |

The most impactful finding was that disabling Qwen3's thinking mode (`/no_think`) had a larger effect on schema validity than any hyperparameter change — think tokens were consuming the token budget before the JSON even started generating.

---

## 7. Bias Detection

**Script:** `Model-Pipeline/Scripts/bias_slicing.py`

The model is evaluated across five demographic and contextual slices to detect performance disparities:

| Slice | Values |
|---|---|
| `goal_type` | strength, mobility, sleep_improvement, longevity, etc. |
| `condition_flag` | has_condition, no_condition |
| `activity_level` | sedentary, lightly_active, moderately_active, very_active |
| `age_band` | various age groups |
| `sex` | M, F, other |

### Metrics Tracked Per Slice
- ROUGE-L mean
- JSON validity rate
- Schema validity rate

### Disparity Detection

Any slice whose schema validity rate deviates more than 15 percentage points from the overall mean is flagged as a potential bias signal.

### Bias Mitigation Strategies

If disparities are detected:
1. **Oversample** underperforming slices in fine-tuning data
2. **Add slice-specific schema examples** to the system prompt
3. **Post-hoc JSON repair** for known failure patterns in specific slices
4. **Re-weighting** loss function to penalize errors on underperforming groups more heavily

Bias reports are saved to `Model-Pipeline/reports/bias_report_<run_id>.json`.

---

## 8. Model Registry

**Script:** `Model-Pipeline/Scripts/push_to_registry.py`

Once the model passes validation and bias checks, the LoRA adapter is:
1. Packaged as a `.tar.gz` archive
2. Uploaded to Google Cloud Storage at `gs://<bucket>/models/fitsense-qwen3-8b/<run_id>/`
3. Registered in Vertex AI Model Registry with eval metrics and bias report attached as metadata

A local `registry_record_<run_id>.json` is saved recording the GCS URI, Vertex resource name, hyperparameters, and push timestamp for full reproducibility.

---

## 9. CI/CD Pipeline

**File:** `.github/workflows/data-pipeline-ci.yml`

The pipeline runs on every push to `main` that affects `Data-Pipeline/` or `Model-Pipeline/`.

### Jobs

| Job | Trigger | What it does |
|---|---|---|
| `test` | Every push/PR | Runs Data Pipeline pytest suite |
| `run-scripts-and-generate-artifacts` | Push to main / manual | Runs synthetic data generation scripts |
| `model-pipeline-validation` | After data pipeline succeeds | Runs schema check, bias slicing, quality gate |

### Quality Gate

The pipeline **fails automatically** if schema validity drops below 50%, blocking any downstream deployment or registry push.

### Rollback Mechanism

Training checkpoints are saved every 20 steps. If a newly trained model performs worse than the previous version, the previous checkpoint can be restored by loading from `checkpoint-40` or `checkpoint-20` in the adapter directory.

---

## 10. Reproduction Steps

### Prerequisites
```bash
pip install unsloth trl transformers peft datasets rouge-score bert-score wandb
pip install google-cloud-aiplatform google-cloud-storage
```

### Step 1 — Prepare training data
```bash
python Model-Pipeline/Scripts/prepare_training_data.py
```

### Step 2 — Train model (requires GPU — run on Google Colab)
```bash
python Model-Pipeline/Scripts/trainmodel.py
```

### Step 3 — Evaluate
```bash
python Model-Pipeline/Scripts/evaluate_student.py
```

### Step 4 — Schema validation
```bash
python Model-Pipeline/Scripts/check_schema.py
```

### Step 5 — Bias analysis
```bash
python Model-Pipeline/Scripts/bias_slicing.py
```

### Step 6 — Push to registry
```bash
python Model-Pipeline/Scripts/push_to_registry.py
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| Qwen3-8B over larger models | Fits on T4 GPU (16GB) with 4-bit quantization |
| LoRA over full fine-tuning | 400x fewer trainable parameters, same schema learning |
| `/no_think` in training | Prevents reasoning tokens from truncating JSON output |
| Explicit schema in system prompt | Eliminates wrong key generation (`workout_plan`, `phases` etc.) |
| `max_new_tokens=2048` | Prevents plan truncation for complex multi-day workouts |
| Checkpointing every 20 steps | Guards against Colab session disconnections |