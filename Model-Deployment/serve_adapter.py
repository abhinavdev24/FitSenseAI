"""Serve LoRA adapter (without merging) over an OpenAI-compatible API.

Loads the 4-bit base model + LoRA adapter using Unsloth (same path as
training) and exposes a /v1/chat/completions endpoint via FastAPI.

Usage:
    python Model-Deployment/serve_adapter.py
    python Model-Deployment/serve_adapter.py --port 8000 --adapter-dir Model-Pipeline/outputs/final_adapter
    python Model-Deployment/serve_adapter.py --base-model unsloth/qwen3-4b-unsloth-bnb-4bit --max-seq-length 16500
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("serve_adapter")

# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "fitsense-adapter"
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    model: str
    choices: list[Choice]
    usage: Usage


# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------

app = FastAPI(title="FitSenseAI Adapter Server")
MODEL = None
TOKENIZER = None
SERVED_MODEL_NAME = "fitsense-adapter"


# ---------------------------------------------------------------------------
# Model loading — mirrors train.py load_model_and_tokenizer
# ---------------------------------------------------------------------------


def load_model(base_model: str, adapter_dir: str, max_seq_length: int):
    """Load base model in 4-bit, apply LoRA adapter, prepare for inference."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    logger.info("Loading base model: %s (4-bit)", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
        fast_inference=False,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    logger.info("Qwen-3 chat template applied")

    logger.info("Loading LoRA adapter from: %s", adapter_dir)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_dir)
    logger.info("Adapter loaded (not merged — running as LoRA)")

    FastLanguageModel.for_inference(model)
    logger.info("Model ready for inference")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(messages: list[dict], max_tokens: int, temperature: float,
             top_p: float, repetition_penalty: float) -> tuple[str, int, int]:
    """Run chat completion and return (text, prompt_tokens, completion_tokens)."""
    input_text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = TOKENIZER(input_text, return_tensors="pt").to(MODEL.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0.01,
            pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
        )

    completion_ids = outputs[0][prompt_len:]
    completion_text = TOKENIZER.decode(completion_ids, skip_special_tokens=True)

    return completion_text, prompt_len, len(completion_ids)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "fitsense"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    start = time.time()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    logger.info("Request: %d messages, max_tokens=%d, temp=%.2f",
                len(messages), req.max_tokens, req.temperature)

    text, prompt_tokens, completion_tokens = generate(
        messages, req.max_tokens, req.temperature,
        req.top_p, req.repetition_penalty,
    )

    elapsed = time.time() - start
    logger.info("Response: %d tokens in %.1fs (%.1f tok/s)",
                completion_tokens, elapsed,
                completion_tokens / max(elapsed, 0.01))

    return ChatCompletionResponse(
        model=SERVED_MODEL_NAME,
        choices=[Choice(message=ChoiceMessage(content=text), finish_reason="stop")],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": SERVED_MODEL_NAME}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Serve LoRA adapter over OpenAI-compatible API")
    parser.add_argument("--base-model", default="unsloth/qwen3-4b-unsloth-bnb-4bit",
                        help="Base model name (default: unsloth/qwen3-4b-unsloth-bnb-4bit)")
    parser.add_argument("--adapter-dir", default="Model-Pipeline/outputs/final_adapter",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--max-seq-length", type=int, default=16500,
                        help="Max sequence length (default: 16500)")
    parser.add_argument("--served-model-name", default="fitsense-adapter",
                        help="Model name in API responses")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser.parse_args()


def main():
    global MODEL, TOKENIZER, SERVED_MODEL_NAME

    args = parse_args()
    SERVED_MODEL_NAME = args.served_model_name

    MODEL, TOKENIZER = load_model(args.base_model, args.adapter_dir, args.max_seq_length)

    logger.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
