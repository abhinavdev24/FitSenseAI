# Model Pipeline — QLoRA SFT with Thinking Distillation

Fine-tuning pipeline for FitSenseAI's student model (Qwen3-4B) using knowledge distilled from a teacher model (Qwen3-32B). The technique is **QLoRA-based Supervised Fine-Tuning (SFT) with thinking distillation** — the student learns both the teacher's reasoning traces (`<think>` blocks) and final JSON tool-call outputs.

---

## Architecture Overview

```mermaid
flowchart TD
    classDef done   fill:#e5e7eb,stroke:#6b7280,color:#374151
    classDef data   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef script fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef output fill:#fef3c7,stroke:#f59e0b,color:#78350f

    A[("responses.jsonl\nTeacher LLM outputs")]:::data
    B["Phase 0 · prepare_training_data.py\n✓ already complete"]:::done
    C[("train.jsonl · val.jsonl\n720 train / 80 val samples")]:::data
    D["Phase 1 · load_data.py\nValidate schema & log stats"]:::script
    E["Phase 3 · hparam_search.py\nOptuna TPE — 10 trials"]:::script
    F[("best_hparams.json")]:::output
    G["Phase 2 · train.py\nQLoRA SFT · 3 epochs · 270 steps"]:::script
    H[("final_adapter/\n63 MB LoRA weights")]:::output
    I["Phase 4 · evaluate.py\ntool accuracy · JSON parse · latency"]:::script
    J["Phase 5 · bias_detection.py\n5 demographic dimensions"]:::script
    K["Phase 6 · sensitivity.py\nhparam + input perturbation"]:::script
    L[("evaluation_results.json\nbias_report.json\nsensitivity_report.json")]:::output
    M["Phase 7 · select_model.py\nN-model composite score"]:::script
    N[("selected_model.json")]:::output
    O["Phase 8 · push_to_registry.py\nGCS upload + versioning"]:::script
    P[("gs://fitsense-models/\nversioned artifact")]:::output

    A --> B --> C --> D
    D --> E --> F --> G --> H
    H --> I & J & K --> L --> M --> N --> O --> P
```

## Training Results

[View run on Weights & Biases](https://wandb.ai/abhinav241998-org/fitsense-sft/runs/zxo4igua?nw=nwuserabhinav241998)

<img src="diagrams/train_loss.png" alt="Train Loss" width="48%"/> <img src="diagrams/eval_loss.png" alt="Eval Loss" width="48%"/>

<img src="diagrams/learning_rate.png" alt="Learning Rate" width="48%"/> <img src="diagrams/train_grad_norm.png" alt="Gradient Norm" width="48%"/>

---

## Training Data Format

Each sample in `train.jsonl` / `val.jsonl` is a 3-turn conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a fitness coaching agent with access to tools..."
    },
    {
      "role": "user",
      "content": "I'm a 28-year-old male, intermediate, looking to build muscle..."
    },
    {
      "role": "assistant",
      "content": "<think>\nThe user wants a hypertrophy program...\n</think>\n{\"tool_name\": \"generate_workout_plan\", \"tool_input\": {...}}"
    }
  ],
  "metadata": {
    "response_id": "...",
    "provider": "groq",
    "has_reasoning": true
  }
}
```

**Dataset stats** (from `prepare_summary.json`):

| Split | Samples | Approx Tokens |
| ----- | ------- | ------------- |
| Train | 720     | ~3.0M         |
| Val   | 80      | ~330K         |

**Reasoning breakdown**: 181 Groq (with thinking) + 558 OpenRouter (with reasoning) + 61 OpenRouter (no reasoning) = 800 total

---

## Framework Stack

```mermaid
graph LR
    classDef hw     fill:#fce7f3,stroke:#db2777,color:#831843
    classDef quant  fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    classDef train  fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef search fill:#fff7ed,stroke:#ea580c,color:#7c2d12
    classDef track  fill:#fef3c7,stroke:#f59e0b,color:#78350f
    classDef cloud  fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f

    GPU["NVIDIA GPU\nCUDA 12.1+"]:::hw
    Torch["PyTorch ≥ 2.1"]:::quant
    BnB["bitsandbytes\nNF4 4-bit quant"]:::quant
    Unsloth["Unsloth ≥ 2024.8\nFast LoRA kernels"]:::train
    PEFT["PEFT ≥ 0.12\nLoRA adapter"]:::train
    TRL["TRL SFTTrainer\n≥ 0.9"]:::train
    Optuna["Optuna ≥ 3.6\nTPE + MedianPruner"]:::search
    WnB["Weights & Biases\nfitsense-sft project"]:::track
    HF["HuggingFace\ndatasets · hub · transformers"]:::cloud
    GCS["Google Cloud Storage\ngs://fitsense-models"]:::cloud

    GPU --> Torch --> BnB --> Unsloth --> PEFT --> TRL
    Optuna -->|best hparams| TRL
    HF -->|base model + dataset| TRL
    TRL -->|metrics + adapter| WnB
    TRL -->|versioned adapter| GCS
```

---

## Directory Structure

```text
Model-Pipeline/
├── scripts/
│   ├── prepare_training_data.py # Phase 0: Converts raw teacher outputs → train/val JSONL
│   ├── load_data.py            # Phase 1: Load & validate train/val JSONL
│   ├── train.py                # Phase 2: QLoRA SFT training loop
│   ├── hparam_search.py        # Phase 3: Bayesian hparam search (Optuna)
│   ├── evaluate.py             # Phase 4: Metrics & generation evaluation
│   ├── bias_detection.py       # Phase 5: Demographic bias analysis
│   ├── sensitivity.py          # Phase 6: Hparam & input sensitivity
│   ├── select_model.py         # Phase 7: Automated model selection
│   └── push_to_registry.py     # Phase 8: GCS upload & versioning
├── config/
│   └── training_config.yaml    # All hyperparameters & paths
├── data/
│   └── training/
│       ├── train.jsonl         # 720 training samples
│       ├── val.jsonl           # 80 validation samples
│       └── prepare_summary.json
├── notebook.ipynb              # Exploratory notebook
├── Dockerfile                  # Multi-stage: training + inference images
├── requirements.txt            # Pinned Python dependencies
├── Plan.md                     # Implementation plan
└── README.md                   # This file
```

**Generated at runtime** (not committed):

```text
Model-Pipeline/
└── outputs/
    ├── final_adapter/              # LoRA weights + tokenizer
    ├── training_summary.json       # Training run metadata
    ├── checkpoints/                # Intermediate checkpoints
    │   ├── checkpoint-220/
    │   ├── checkpoint-230/
    │   ├── checkpoint-240/
    │   ├── checkpoint-250/
    │   └── checkpoint-260/
    ├── hparam_search/
    │   ├── best_hparams.json       # Winning hyperparameters
    │   └── all_trials.json         # All Optuna trial results
    ├── evaluation/
    │   ├── evaluation_results.json # Aggregate metrics
    │   ├── per_sample_results.jsonl
    │   └── plots/                  # Metric visualizations
    ├── bias_detection/
    │   ├── bias_report.json        # Flagged slices & recommendations
    │   └── plots/                  # Bias heatmaps
    ├── sensitivity/
    │   ├── sensitivity_report.json
    │   └── plots/
    ├── selection/
    │   └── selected_model.json     # Final model decision
    └── registry/
        └── {model_name}/{version}/ # Staged package for GCS
```

---

## Prerequisites

- **Python 3.11+** (via conda)
- **NVIDIA GPU** with CUDA 12.1+ (for training/evaluation)
- **Conda environment**: `mlopsenv`
- **Weights & Biases** account (for experiment tracking)
- **GCP credentials** (for registry push)

```bash
# Create and activate conda environment
conda create -n mlopsenv python=3.11 -y
conda activate mlopsenv

# Install dependencies
pip install -r Model-Pipeline/requirements.txt
```

### Environment Variables

| Variable                         | Purpose                               | Required For            |
| -------------------------------- | ------------------------------------- | ----------------------- |
| `WANDB_API_KEY`                  | Weights & Biases experiment tracking  | Training, hparam search |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account key path          | Registry push           |
| `GROQ_API_KEY`                   | Teacher LLM calls (upstream pipeline) | Data preparation only   |

---

## Pipeline Execution

### Quick Reference

```text
 Phase 0   prepare_training_data.py    ── already done ──▶  train.jsonl + val.jsonl
 Phase 1   load_data.py                ── validate data  ──▶  HF Dataset
 Phase 3   hparam_search.py            ── Optuna search  ──▶  best_hparams.json
 Phase 2   train.py                    ── QLoRA SFT      ──▶  LoRA adapter
 Phase 4   evaluate.py                 ── metrics + gen   ──▶  evaluation_results.json
 Phase 5   bias_detection.py           ── slice analysis  ──▶  bias_report.json
 Phase 6   sensitivity.py              ── perturbation    ──▶  sensitivity_report.json
 Phase 7   select_model.py             ── compare models  ──▶  selected_model.json
 Phase 8   push_to_registry.py         ── GCS upload      ──▶  Artifact Registry
```

> **Note**: Always run from the repository root and with `conda activate mlopsenv`.

---

### Phase 0: Prepare Training Data (already complete)

Converts raw teacher LLM responses into the messages format for SFT.

```bash
python Model-Pipeline/scripts/prepare_training_data.py \
    --input  Data-Pipeline/data/raw/teacher-llm-responses/20260324T162857Z/responses.jsonl \
    --output Model-Pipeline/data/training \
    --val-ratio 0.1 \
    --seed 42
```

### Phase 1: Load & Validate Data

Loads the JSONL files, validates schema (3 messages per row, correct roles), and logs stats.

```bash
python Model-Pipeline/scripts/load_data.py \
    --train-path Model-Pipeline/data/training/train.jsonl \
    --val-path   Model-Pipeline/data/training/val.jsonl
```

This phase is also callable from other scripts:

```python
from load_data import load_and_validate
datasets = load_and_validate("Model-Pipeline/data/training/train.jsonl",
                              "Model-Pipeline/data/training/val.jsonl")
```

### Phase 2: Train (QLoRA SFT)

Trains a LoRA adapter on the 4-bit quantized base model.

```bash
python Model-Pipeline/scripts/train.py \
    --config Model-Pipeline/config/training_config.yaml
```

See [Configuration Reference](#configuration-reference) for all parameters.

**Output**: `outputs/final_adapter/` (~63MB), `outputs/training_summary.json`

### Phase 3: Hyperparameter Search

Bayesian optimization with Optuna to find the best training configuration.

```bash
python Model-Pipeline/scripts/hparam_search.py \
    --config Model-Pipeline/config/training_config.yaml \
    --n-trials 10 \
    --output-dir Model-Pipeline/outputs/hparam_search
```

**Search space**:

| Hyperparameter  | Range             | Type        |
| --------------- | ----------------- | ----------- |
| `lora_r`        | [8, 16, 32]       | Categorical |
| `lora_alpha`    | 2 x `lora_r`      | Derived     |
| `learning_rate` | [1e-4, 5e-4]      | Log-uniform |
| `batch_size`    | [1]               | Fixed       |
| `lora_dropout`  | [0.0, 0.05, 0.1]  | Categorical |
| `warmup_ratio`  | [0.03, 0.05, 0.1] | Categorical |

Uses `TPESampler` (Tree-structured Parzen Estimator) with `MedianPruner` for early stopping of underperforming trials.

**Trial results** (10 Optuna trials, sorted by train loss) — [View on Weights & Biases](https://wandb.ai/abhinav241998-org/fitsense-sft/runs/2m5v6n0l?nw=nwuserabhinav241998):

| Trial | Train Loss | LoRA r | Learning Rate | Batch Size | Dropout | Warmup | State    |
| ----- | ---------- | ------ | ------------- | ---------- | ------- | ------ | -------- |
| 9     | 0.3611     | 8      | 3.460e-4      | 1          | 0.00    | 0.05   | COMPLETE |
| 8     | 0.3678     | 32     | 1.255e-4      | 1          | 0.10    | 0.10   | COMPLETE |
| 6     | 0.3757     | 16     | 1.651e-4      | 1          | 0.00    | 0.05   | COMPLETE |
| 3     | 0.3774     | 16     | 1.632e-4      | 1          | 0.00    | 0.03   | COMPLETE |
| 2     | 0.4297     | 16     | 1.098e-4      | 1          | 0.00    | 0.03   | COMPLETE |
| 5     | 0.4375     | 8      | 1.633e-4      | 1          | 0.05    | 0.05   | COMPLETE |
| 4     | 0.4683     | 8      | 1.379e-4      | 1          | 0.10    | 0.05   | COMPLETE |

Best config: `lora_r=8`, `lr=3.46e-4`, `dropout=0`, `warmup=0.05` (Trial 9, loss 0.3611)

**Output**: `best_hparams.json`, `all_trials.json`

### Phase 4: Evaluation

Generates responses on the validation set and computes metrics.

```bash
python Model-Pipeline/scripts/evaluate.py \
    --adapter-dir Model-Pipeline/outputs/final_adapter \
    --config Model-Pipeline/config/training_config.yaml \
    --output-dir Model-Pipeline/outputs/evaluation \
    --max-samples 50  # optional: quick check
```

**Metrics computed**:

| Metric                 | What It Measures                             |
| ---------------------- | -------------------------------------------- |
| Validation Loss        | Cross-entropy on held-out set (forward pass) |
| Tool Call Accuracy     | Correct `tool_name` prediction rate          |
| JSON Parse Rate        | % of outputs that are valid JSON             |
| Schema Compliance      | % with `tool_name` + `tool_input` keys       |
| Thinking Presence Rate | % of responses with `<think>` block          |
| Avg Thinking Length    | Avg characters in reasoning traces           |
| Response Latency       | Avg ms per `model.generate()` call           |

Also computes per-tool accuracy breakdown and generates plots (bar charts, latency histogram).

**Output**: `evaluation_results.json`, `per_sample_results.jsonl`, `plots/`

### Phase 5: Bias Detection

Slices validation results by demographic attributes and flags disparities.

```bash
python Model-Pipeline/scripts/bias_detection.py \
    --adapter-dir Model-Pipeline/outputs/final_adapter \
    --config Model-Pipeline/config/training_config.yaml \
    --output-dir Model-Pipeline/outputs/bias_detection \
    --threshold 0.1
```

**Slicing dimensions** (extracted from user messages via regex):

```mermaid
graph TD
    classDef root   fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef dim    fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef slice  fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef metric fill:#fef3c7,stroke:#f59e0b,color:#78350f

    VD["Validation Set\n80 samples"]:::root

    VD --> AG["Age Group"]:::dim
    VD --> GD["Gender"]:::dim
    VD --> FL["Fitness Level"]:::dim
    VD --> GT["Goal Type"]:::dim
    VD --> BM["BMI Category"]:::dim

    AG --> AG1["18–25"]:::slice
    AG --> AG2["26–35"]:::slice
    AG --> AG3["36–50"]:::slice
    AG --> AG4["50+"]:::slice

    GD --> GD1["Male"]:::slice
    GD --> GD2["Female"]:::slice

    FL --> FL1["Beginner"]:::slice
    FL --> FL2["Intermediate"]:::slice
    FL --> FL3["Advanced"]:::slice

    GT --> GT1["Weight Loss"]:::slice
    GT --> GT2["Muscle Gain"]:::slice
    GT --> GT3["Endurance"]:::slice

    BM --> BM1["Underweight"]:::slice
    BM --> BM2["Normal"]:::slice
    BM --> BM3["Overweight"]:::slice
    BM --> BM4["Obese"]:::slice

    AG1 & AG2 & AG3 & AG4 & GD1 & GD2 & FL1 & FL2 & FL3 & GT1 & GT2 & GT3 & BM1 & BM2 & BM3 & BM4 --> M

    M["Per-slice metrics\ntool_call_accuracy · json_parse_rate\nschema_compliance · thinking_presence_rate\n⚠ flag if deviation > 10%"]:::metric
```

Generates heatmap visualizations and mitigation recommendations.

**Output**: `bias_report.json`, `plots/bias_heatmap.png`

### Phase 6: Sensitivity Analysis

Analyzes how performance changes with hyperparameters and input perturbations.

```bash
python Model-Pipeline/scripts/sensitivity.py \
    --adapter-dir Model-Pipeline/outputs/final_adapter \
    --config Model-Pipeline/config/training_config.yaml \
    --trials-file Model-Pipeline/outputs/hparam_search/all_trials.json \
    --output-dir Model-Pipeline/outputs/sensitivity \
    --n-samples 50
```

**Two analysis types**:

1. **Hyperparameter sensitivity** — which hparams affect val loss the most (from Optuna trials)
2. **Input perturbation** — 4 tests:
   - Truncate user message to 50% / 25%
   - Remove system prompt entirely
   - Mask demographic info (age, gender, BMI, fitness level)

Use `--skip-hparam` or `--skip-input` to run only one type.

**Output**: `sensitivity_report.json`, hparam ranking plots, perturbation impact chart

### Phase 7: Model Selection

Compares the final adapter against all saved checkpoints and selects the best one using a weighted composite score.

```bash
python Model-Pipeline/scripts/select_model.py \
    --eval-dirs \
        Model-Pipeline/outputs/evaluation/final_adapter \
        Model-Pipeline/outputs/evaluation/checkpoint-220 \
        Model-Pipeline/outputs/evaluation/checkpoint-230 \
        Model-Pipeline/outputs/evaluation/checkpoint-240 \
        Model-Pipeline/outputs/evaluation/checkpoint-250 \
        Model-Pipeline/outputs/evaluation/checkpoint-260 \
    --bias-dirs \
        Model-Pipeline/outputs/bias_detection/final_adapter \
        Model-Pipeline/outputs/bias_detection/checkpoint-220 \
        Model-Pipeline/outputs/bias_detection/checkpoint-230 \
        Model-Pipeline/outputs/bias_detection/checkpoint-240 \
        Model-Pipeline/outputs/bias_detection/checkpoint-250 \
        Model-Pipeline/outputs/bias_detection/checkpoint-260 \
    --output-dir Model-Pipeline/outputs/selection
```

**Scoring weights**:

```text
 Tool Call Accuracy ████████████████████████████████  0.30
 JSON Parse Rate    ████████████████████             0.20
 Schema Compliance  ███████████████                  0.15
 Val Loss (inv.)    ███████████████                  0.15
 Thinking Presence  ██████████                       0.10
 Bias Score (inv.)  ██████████                       0.10
```

**Eval → Select → Push flow**:

```mermaid
flowchart TD
    classDef ckpt   fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef eval   fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef bias   fill:#fce7f3,stroke:#db2777,color:#831843
    classDef select fill:#fef3c7,stroke:#f59e0b,color:#78350f
    classDef push   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f

    subgraph CANDS["6 Candidates (outputs/checkpoints/ + final_adapter)"]
        direction LR
        FA["final_adapter"]:::ckpt
        C1["checkpoint-220"]:::ckpt
        C2["checkpoint-230"]:::ckpt
        C3["checkpoint-240"]:::ckpt
        C4["checkpoint-250"]:::ckpt
        C5["checkpoint-260"]:::ckpt
    end

    CANDS -->|"evaluate.py\n--output-dir outputs/evaluation/&lt;name&gt;"| EV["evaluation_results.json\nper candidate"]:::eval
    CANDS -->|"bias_detection.py\n--output-dir outputs/bias_detection/&lt;name&gt;"| BI["bias_report.json\nper candidate"]:::bias

    EV & BI --> SM["select_model.py\n--eval-dirs ... --bias-dirs ...\nweighted composite score"]:::select
    SM --> BEST["selected_model.json\nbest candidate path + rationale"]:::select
    BEST --> PR["push_to_registry.py\n--adapter-dir &lt;selected&gt;\n--config registry_config.yaml"]:::push
    PR --> GCS["gs://fitsense-models/\nversioned upload"]:::push
```

**Output**: `selected_model.json` with scores, breakdown, rationale, and comparison table

### Phase 8: Push to Registry

Packages the adapter + all training checkpoints and uploads to GCS and HuggingFace Hub.

```bash
# Full upload — adapter + checkpoints + metadata → GCS + HuggingFace
conda run -n mlopsenv python Model-Pipeline/scripts/push_to_registry.py \
    --adapter-dir      Model-Pipeline/outputs/final_adapter \
    --checkpoints-dir  Model-Pipeline/outputs/checkpoints \
    --metadata-files \
        Model-Pipeline/outputs/training_summary.json \
        Model-Pipeline/outputs/hparam_search/best_hparams.json \
        Model-Pipeline/outputs/evaluation/evaluation_results.json \
        Model-Pipeline/outputs/bias_detection/bias_report.json \
        Model-Pipeline/outputs/sensitivity/sensitivity_report.json \
    --gcs-bucket  fitsense-models \
    --model-name  qwen3-4b-fitsense-qlora \
    --hf-repo     abhinav241998/qwen3-4b-fitsense-qlora \
    --hf-token    $HF_TOKEN \
    --log-level   INFO

# GCS only (no HuggingFace push)
conda run -n mlopsenv python Model-Pipeline/scripts/push_to_registry.py \
    --adapter-dir     Model-Pipeline/outputs/final_adapter \
    --checkpoints-dir Model-Pipeline/outputs/checkpoints \
    --metadata-files  Model-Pipeline/outputs/training_summary.json \
    --gcs-bucket      fitsense-models \
    --model-name      qwen3-4b-fitsense-qlora \
    --log-level       INFO

# Dry run (local staging only, no uploads)
conda run -n mlopsenv python Model-Pipeline/scripts/push_to_registry.py \
    --adapter-dir Model-Pipeline/outputs/final_adapter \
    --gcs-bucket  fitsense-models \
    --model-name  qwen3-4b-fitsense-qlora \
    --dry-run

# Rollback latest.json to a previous version
conda run -n mlopsenv python Model-Pipeline/scripts/push_to_registry.py \
    --adapter-dir  Model-Pipeline/outputs/final_adapter \
    --gcs-bucket   fitsense-models \
    --model-name   qwen3-4b-fitsense-qlora \
    --rollback-to  v20260324T120000Z
```

**GCS layout** (`gs://fitsense-models/`):

```text
gs://fitsense-models/
└── qwen3-4b-fitsense-qlora/
    ├── latest.json                        # Points to current version
    ├── versions.json                      # Version history
    └── v<timestamp>/
        ├── manifest.json
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── tokenizer.json
        ├── training_summary.json
        └── checkpoints/
            ├── checkpoint-220/
            ├── checkpoint-230/
            └── ...
```

**Model versioning & rollback flow**:

```mermaid
flowchart LR
    classDef action fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef vers   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef ptr    fill:#fef3c7,stroke:#f59e0b,color:#78350f

    P1["push run 1\nv20260325T..."]:::action --> V1["v20260325T.../\nadapter + metadata"]:::vers
    P2["push run 2\nv20260401T..."]:::action --> V2["v20260401T.../\nadapter + metadata"]:::vers
    P3["push run 3\nv20260411T..."]:::action --> V3["v20260411T... ← current\nadapter + metadata"]:::vers

    V1 & V2 & V3 --> VJ["versions.json\n[v20260325T, v20260401T, v20260411T]"]:::ptr
    V3 --> LJ["latest.json\n{ version: v20260411T }"]:::ptr

    RB["--rollback-to v20260401T\nupdates latest.json only\nno data deleted"]:::action --> LJ
```

**HuggingFace Hub**: [abhinav241998/qwen3-4b-fitsense-qlora](https://huggingface.co/abhinav241998/qwen3-4b-fitsense-qlora)

- Base model: `unsloth/qwen3-4b-unsloth-bnb-4bit`
- Adapter pushed as a PEFT LoRA repo with version tag matching GCS

---

## CI/CD Pipeline

Automated via GitHub Actions (`.github/workflows/model_pipeline.yml`). The `lint-and-test` job runs automatically on every push or PR to `Model-Pipeline/**`. All heavy GPU jobs (train, evaluate, bias-check, registry-push) require a manual `workflow_dispatch` trigger.

```mermaid
flowchart TD
    classDef trigger fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef auto    fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef manual  fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef gate    fill:#fef3c7,stroke:#f59e0b,color:#78350f
    classDef fail    fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef success fill:#ecfdf5,stroke:#059669,color:#064e3b

    PR["Push / PR\nto Model-Pipeline/**"]:::trigger
    WD["workflow_dispatch\nManual trigger\nmodel_name · skip_training"]:::trigger

    L["lint-and-test\nubuntu-latest\nruff · pyright · pytest"]:::auto
    T["train\nGPU self-hosted\nhparam_search → train.py"]:::manual
    E["evaluate\nGPU self-hosted\nevaluate.py"]:::manual
    BC["bias-check\nGPU self-hosted\nbias_detection.py"]:::manual
    BG{"deviation\n> 25%?"}:::gate
    RP["registry-push\nmain branch only\n+ manual approval gate"]:::manual
    NF["notify\nSlack webhook\nsuccess · fail · blocked"]:::success
    BLOCKED["Pipeline BLOCKED\ncritical bias detected"]:::fail

    PR --> L
    WD --> L
    L -->|"workflow_dispatch only"| T --> E --> BC --> BG
    BG -->|No| RP --> NF
    BG -->|Yes| BLOCKED --> NF
    L -->|"push / PR only\n(stops here)"| NF
```

**Key features**:

- **Manual approval gate**: `registry-push` uses the `production` environment (requires reviewer approval)
- **Critical bias blocking**: Pipeline fails if any slice has >25% deviation
- **Slack notifications**: Success, failure, and blocked states with color coding
- **Workflow dispatch**: Manual trigger with model name override and skip-training option

**Required GitHub Secrets**:

| Secret              | Purpose                           |
| ------------------- | --------------------------------- |
| `WANDB_API_KEY`     | Experiment tracking               |
| `GCP_SA_KEY`        | GCP service account JSON          |
| `SLACK_WEBHOOK_URL` | Pipeline notifications (optional) |

---

## GCP Deployment Architecture

```mermaid
graph TB
    classDef gh    fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef local fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef gcs   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef hf    fill:#ede9fe,stroke:#7c3aed,color:#4c1d95

    GHA["GitHub Actions\n.github/workflows/model_pipeline.yml"]:::gh
    GPU["Self-hosted GPU Runner\nNVIDIA + CUDA 12.1"]:::local

    subgraph TRAIN["Training (self-hosted GPU)"]
        direction TB
        HS["hparam_search.py\n10 Optuna trials"]:::local
        TR["train.py\nQLoRA SFT"]:::local
        EV["evaluate.py + bias_detection.py\n+ sensitivity.py"]:::local
        SM["select_model.py\nN-checkpoint comparison"]:::local
    end

    subgraph GCS_BUCKET["gs://fitsense-models/qwen3-4b-fitsense-qlora/"]
        direction TB
        LATEST["latest.json\n→ current version pointer"]:::gcs
        VERSIONS["versions.json\nfull version history"]:::gcs
        subgraph VX["v&lt;timestamp&gt;/"]
            direction LR
            F1["manifest.json\nadapter_config.json\nadapter_model.safetensors"]:::gcs
            F2["tokenizer.json\ntraining_summary.json\nevaluation_results.json\nbias_report.json"]:::gcs
            F3["checkpoints/\ncheckpoint-220/ · checkpoint-230/\ncheckpoint-240/ · checkpoint-250/\ncheckpoint-260/"]:::gcs
        end
    end

    HF["HuggingFace Hub\nabhinav241998/qwen3-4b-fitsense-qlora\n+ version tag"]:::hf

    GHA -->|"triggers (workflow_dispatch)"| GPU
    GPU --> TRAIN
    TRAIN -->|"push_to_registry.py"| GCS_BUCKET
    TRAIN -->|"adapter + tag"| HF
    F1 --> LATEST
    F1 --> VERSIONS
```

---

## Docker

Multi-stage build with two targets:

```bash
# Build training image (CUDA devel, full deps)
docker build --target training -t fitsense-train -f Model-Pipeline/Dockerfile .

# Build inference image (CUDA runtime, lightweight)
docker build --target inference -t fitsense-eval -f Model-Pipeline/Dockerfile .

# Run training (mount data + pass W&B key)
docker run --gpus all \
    -v $(pwd)/Model-Pipeline/data:/app/Model-Pipeline/data \
    -v $(pwd)/Model-Pipeline/outputs:/app/Model-Pipeline/outputs \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    fitsense-train \
    --config Model-Pipeline/config/training_config.yaml

# Run evaluation
docker run --gpus all \
    -v $(pwd)/Model-Pipeline/outputs:/app/Model-Pipeline/outputs \
    -v $(pwd)/Model-Pipeline/data:/app/Model-Pipeline/data \
    fitsense-eval \
    --adapter-dir Model-Pipeline/outputs/final_adapter \
    --config Model-Pipeline/config/training_config.yaml
```

| Stage       | Base Image                               | Size | Purpose                                |
| ----------- | ---------------------------------------- | ---- | -------------------------------------- |
| `training`  | `nvidia/cuda:12.1.1-devel-ubuntu22.04`   | ~8GB | Full training with compilation support |
| `inference` | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | ~5GB | Evaluation and inference only          |

---

## Experiment Tracking

All training and search experiments are logged to **Weights & Biases** under the `fitsense-sft` project.

**What gets logged**:

- `train.py`: Loss curves, learning rate, GPU utilization, final adapter artifact
- `hparam_search.py`: Single summary run with all-trials table + best params
- Eval/bias results uploaded as artifacts alongside the adapter

Set your API key:

```bash
export WANDB_API_KEY=your_key_here
# or
wandb login
```

---

## Configuration Reference

All training parameters live in `config/training_config.yaml`:

```yaml
# Model
model_name: "unsloth/qwen3-4b-unsloth-bnb-4bit" # Base model from HuggingFace
max_seq_length: 16500 # Max tokens per sequence

# LoRA
lora_r: 8 # Adapter rank
lora_alpha: 16 # Scaling factor (2x rank)
lora_dropout: 0.05 # Regularization

# Training
batch_size: 1 # Per-device batch size
gradient_accumulation_steps: 8 # Effective batch = 8
num_epochs: 3 # Training epochs
learning_rate: 0.000346015 # Peak learning rate (from best_hparams.json)
lr_scheduler_type: "cosine" # LR decay schedule
warmup_ratio: 0.05 # % of steps for warmup

# Tracking
report_to: "wandb" # "wandb", "mlflow", or "none"
wandb_project: "fitsense-sft" # W&B project name
```

Override any value via CLI: `--model-name unsloth/Qwen3-4B` or `--output-dir /custom/path`
