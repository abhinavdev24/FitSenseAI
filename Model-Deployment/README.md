# Model-Deployment

Scripts for serving and evaluating the fine-tuned FitSenseAI model.

## Scripts

### `serve_adapter.py` — Serve LoRA adapter

Loads the 4-bit base model with a LoRA adapter (via Unsloth) and exposes an OpenAI-compatible REST API using FastAPI + Uvicorn. The adapter is **not merged** — it runs as a live LoRA on top of the quantized base.

**Endpoints:**

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `GET` | `/v1/models` | List served models |
| `GET` | `/health` | Health check |

**Usage:**

```bash
# Default: loads adapter from Model-Pipeline/outputs/final_adapter, port 8000
python Model-Deployment/serve_adapter.py

# Custom adapter and base model
python Model-Deployment/serve_adapter.py \
  --base-model unsloth/qwen3-4b-unsloth-bnb-4bit \
  --adapter-dir Model-Pipeline/outputs/final_adapter \
  --max-seq-length 16500 \
  --served-model-name fitsense-adapter \
  --host 0.0.0.0 \
  --port 8000
```

**Arguments:**

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--base-model` | `unsloth/qwen3-4b-unsloth-bnb-4bit` | HuggingFace base model name |
| `--adapter-dir` | `Model-Pipeline/outputs/final_adapter` | Path to LoRA adapter directory |
| `--max-seq-length` | `16500` | Maximum sequence length |
| `--served-model-name` | `fitsense-adapter` | Model name returned in API responses |
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8000` | Bind port |

---

### `eval_vllm.py` — Evaluate against a vLLM server

Samples queries from the training JSONL, sends them to a running vLLM (or `serve_adapter.py`) server, and saves the responses alongside ground-truth answers for offline inspection.

**Usage:**

```bash
# Default: 5 random queries from train.jsonl → eval_results.jsonl
python Model-Deployment/eval_vllm.py

# Custom run
python Model-Deployment/eval_vllm.py \
  --data Model-Pipeline/data/training/train.jsonl \
  --num-queries 20 \
  --model fitsense-bf16 \
  --base-url http://localhost:8000 \
  --max-tokens 16000 \
  --temperature 0.7 \
  --repetition-penalty 1.0 \
  --output Model-Deployment/eval_results.jsonl \
  --seed 42
```

**Arguments:**

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--data` | `Model-Pipeline/data/training/train.jsonl` | Path to JSONL evaluation data |
| `--num-queries` | `5` | Number of queries to sample and send |
| `--model` | `fitsense-bf16` | Served model name (must match server) |
| `--base-url` | `http://localhost:8000` | vLLM server base URL |
| `--max-tokens` | `16000` | Max completion tokens |
| `--temperature` | `0.7` | Sampling temperature |
| `--repetition-penalty` | `1.0` | Repetition penalty (`1.0` = off) |
| `--no-think` | — | Append `/no_think` to system prompt (disables chain-of-thought) |
| `--no-fences` | — | Instruct the model to output raw JSON without markdown fences |
| `--output` | `Model-Deployment/eval_results.jsonl` | Path to write results |
| `--seed` | `42` | Random seed for query shuffling |

**Output format** (`eval_results.jsonl`): each line is a JSON record with:

- `query_index`, `user_message`, `ground_truth`, `model_response`
- `finish_reason`, `usage` (prompt/completion/total tokens), `elapsed_seconds`
- On error: `error` field instead of response fields

---

## Typical workflow

```bash
# 1. Start the adapter server
python Model-Deployment/serve_adapter.py

# 2. In another terminal, run evaluation
python Model-Deployment/eval_vllm.py --num-queries 20 --model fitsense-adapter

# 3. Inspect results
cat Model-Deployment/eval_results.jsonl | python -m json.tool | less
```

> For vLLM-served merged weights (e.g. `fitsense-bf16`), point `--base-url` at the vLLM process and set `--model` to match the name it was launched with.
