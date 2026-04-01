# FitSenseAI — Model Pipeline Implementation Plan

Last updated: 2026-03-05

## Context

The assignment (Model_pipeline.pdf) requires building the ML model development layer on top of the already-completed data pipeline. The data pipeline (Phases 1–6 in `Data-Pipeline/`) already produces train/val/test JSONL distillation datasets and QA reports. What's missing is the entire model development phase: fine-tuning a student model, experiment tracking, hyperparameter tuning, model validation, sensitivity analysis, bias detection, and CI/CD automation.

**Choices made:**

- Student model: Qwen3 8B (LoRA/QLoRA fine-tuning)
- Experiment tracking: Weights & Biases
- Registry: GCS-based JSON manifest (local-first, push to GCS)

---

## Assignment Requirements Checklist

From `assignment_pdf/Model_pipeline.pdf`:

| Requirement                        | Implementation                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| Load data from data pipeline       | `model/train.py` reads `Data-Pipeline/data/raw/distillation_dataset/<run_id>/` |
| Train + select best model          | LoRA fine-tuning + W&B comparison                                              |
| Model validation (accuracy/F1/AUC) | `model/evaluate.py` on held-out test.jsonl                                     |
| Bias detection (slicing)           | `model/bias_detection.py` using slice_tags from distillation dataset           |
| Bias check code                    | Fairlearn-style sliced metrics + report                                        |
| Push to Artifact/Model Registry    | `model/registry.py` → GCS manifest                                             |
| Hyperparameter tuning              | `model/hparam_search.py` — random search                                       |
| Experiment tracking (W&B)          | Log all runs, hyperparams, metrics, model versions                             |
| Results visualizations             | Bar plots, metric comparisons via W&B + local artifacts                        |
| Sensitivity analysis               | `model/sensitivity.py` — feature importance + HP sensitivity plots             |
| CI/CD pipeline                     | `.github/workflows/model_pipeline.yml`                                         |
| Docker/containerization            | `model/Dockerfile`                                                             |
| Notifications/alerts               | W&B alerts + GitHub Actions failure notifications                              |
| Rollback mechanism                 | Registry manifest stores previous model version                                |

---

## New Directory Structure

```
model/
  config.yaml               # training config: model, LoRA params, dataset run_id
  requirements.txt          # transformers, peft, trl, wandb, fairlearn, captum
  Dockerfile                # containerized training entrypoint
  train.py                  # main fine-tuning script
  evaluate.py               # validation metrics on val.jsonl / test.jsonl
  hparam_search.py          # random search over LR, LoRA rank, epochs
  sensitivity.py            # feature importance (token attribution) + HP sensitivity
  bias_detection.py         # sliced evaluation across slice_tags + bias report
  registry.py               # push model + manifest to GCS / local store

.github/
  workflows/
    model_pipeline.yml      # CI/CD: train → validate → bias check → push registry
```

---

## Existing Files to Reuse

- `Data-Pipeline/data/raw/distillation_dataset/<run_id>/train.jsonl` — training data
- `Data-Pipeline/data/raw/distillation_dataset/<run_id>/val.jsonl` — validation data
- `Data-Pipeline/data/raw/distillation_dataset/<run_id>/test.jsonl` — held-out test
- `Data-Pipeline/params.yaml` — reuse `run_id`, seed, slice tag definitions
- `Data-Pipeline/scripts/common/` — shared utilities (logging, config loading)

---

## Implementation Steps

### Step 1 — `model/config.yaml`

Defines all tunable parameters:

- `model_name: "Qwen/Qwen3-8B"`
- `dataset_run_id: <latest run_id from Data-Pipeline>`
- `dataset_path: "Data-Pipeline/data/raw/distillation_dataset"`
- LoRA params: `r`, `lora_alpha`, `lora_dropout`, `target_modules`
- Training: `lr`, `epochs`, `batch_size`, `max_seq_len`, `output_dir`
- W&B: `project`, `entity`
- Registry: `gcs_bucket` or `local_registry_path`

### Step 2 — `model/train.py`

1. Load config from `config.yaml`
2. Load `train.jsonl` and `val.jsonl` from distillation dataset
3. Initialize W&B run, log config
4. Load Qwen3-8B via `transformers`, apply LoRA via `peft` (QLoRA with 4-bit via `bitsandbytes`)
5. Fine-tune with `trl.SFTTrainer`
6. Evaluate on `val.jsonl` each epoch, log metrics to W&B
7. Save best checkpoint based on val loss / ROUGE / BERTScore
8. Log model artifact to W&B

### Step 3 — `model/evaluate.py`

1. Load a trained model checkpoint
2. Run inference on `test.jsonl`
3. Compute metrics: ROUGE-1/2/L, BERTScore, perplexity, response coherence
4. For classification sub-tasks: accuracy, precision, recall, F1
5. Generate bar plots and confusion matrices, save to `model/artifacts/`
6. Log metrics + plots to W&B
7. Output a `eval_report.json`

### Step 4 — `model/hparam_search.py`

1. Define search space: `lr` (1e-5 to 5e-4), `lora_r` (8/16/32), `epochs` (1/2/3), `batch_size` (4/8)
2. Run N random configurations (default N=6)
3. Each run: call `train.py` logic, log to W&B with `wandb.sweep`
4. Compare via W&B runs table, select best config
5. Save best hyperparameters back to `config.yaml`

### Step 5 — `model/sensitivity.py`

1. **Feature/token sensitivity**: For held-out prompts, compute gradient-based attribution scores over input tokens (via `captum` integrated gradients). Identify which prompt parts (user profile vs. query text) most influence outputs.
2. **Hyperparameter sensitivity**: Plot val loss vs. each HP dimension across sweep runs (via W&B API). Generate bar charts showing which HPs have the highest impact.
3. Save plots to `model/artifacts/sensitivity/`

### Step 6 — `model/bias_detection.py`

1. Load `test.jsonl` which includes `slice_tags` (age_group, fitness_level, goal_type, gender) from the distillation dataset
2. Run model inference on each record
3. Compute metrics (ROUGE, perplexity, response length, safety flags) **per slice**
4. Use `fairlearn` `MetricFrame` to compute disparities across slices
5. Generate:
   - Per-slice metric table
   - Bar chart of metric disparities
   - `bias_report.json` with pass/fail per slice
6. Flag if any slice has >10% disparity vs. overall mean (configurable threshold)
7. Log results to W&B

### Step 7 — `model/registry.py`

1. Collect model artifact path, `eval_report.json`, `bias_report.json`
2. Build version manifest JSON:
   ```json
   {
     "model_version": "<timestamp>-<git-sha>",
     "model_path": "<gcs_uri or local_path>",
     "dataset_run_id": "<run_id>",
     "git_sha": "<commit>",
     "val_metrics": {...},
     "test_metrics": {...},
     "bias_passed": true,
     "wandb_run_url": "...",
     "registered_at": "<ISO timestamp>"
   }
   ```
3. Store manifest at `model/registry/manifest.json` (local) and optionally push to GCS
4. Keep a `registry/history.json` log of all past versions (for rollback)

### Step 8 — `model/Dockerfile`

- Base: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
- Install `model/requirements.txt`
- Entrypoint: `python train.py --config config.yaml`
- Supports `--mode evaluate`, `--mode bias_detection` via arg dispatch

### Step 9 — `.github/workflows/model_pipeline.yml`

Triggers on push to `main` when any file in `model/` changes:

```
jobs:
  model-pipeline:
    steps:
      1. Checkout + setup Python
      2. Install model/requirements.txt
      3. Run model/train.py (using latest dataset run_id)
      4. Run model/evaluate.py → check val metrics threshold
      5. Run model/bias_detection.py → check bias_passed
      6. If both pass: run model/registry.py (push manifest)
      7. On failure: alert via W&B + GitHub Actions failure notification
      8. Rollback: if new model fails validation, registry retains previous version
```

---

## Key Dependencies (`model/requirements.txt`)

```
transformers>=4.45
peft>=0.12
trl>=0.11
bitsandbytes>=0.43
torch>=2.3
wandb>=0.17
fairlearn>=0.10
rouge-score
bert-score
captum
google-cloud-storage
```

---

## Verification Plan

1. **Local smoke test**: `python model/train.py --config model/config.yaml --dry-run` (small subset, 1 epoch)
2. **Evaluate**: `python model/evaluate.py` → check `eval_report.json` has all metrics
3. **Bias**: `python model/bias_detection.py` → check `bias_report.json` has all slice_tags covered
4. **Sensitivity**: `python model/sensitivity.py` → confirm plots in `model/artifacts/sensitivity/`
5. **Registry**: `python model/registry.py` → check `model/registry/manifest.json` and `history.json`
6. **W&B**: Verify W&B dashboard shows runs with logged metrics, plots, and model artifacts
7. **CI/CD**: Push a change to `model/` → verify GitHub Actions pipeline runs end-to-end

---

## Critical Files

| File                                           | Purpose                           |
| ---------------------------------------------- | --------------------------------- |
| `model/config.yaml`                            | All training and pipeline config  |
| `model/train.py`                               | LoRA fine-tuning of Qwen3-8B      |
| `model/evaluate.py`                            | Validation + test metrics         |
| `model/hparam_search.py`                       | Hyperparameter random search      |
| `model/sensitivity.py`                         | Feature + HP sensitivity analysis |
| `model/bias_detection.py`                      | Fairlearn sliced bias evaluation  |
| `model/registry.py`                            | GCS/local model version manifest  |
| `model/Dockerfile`                             | Containerized training            |
| `.github/workflows/model_pipeline.yml`         | CI/CD automation                  |
| `Data-Pipeline/data/raw/distillation_dataset/` | Input data (existing)             |
