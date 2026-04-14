"""Send sample queries from train.jsonl to a vLLM server and save responses."""

import argparse
import json
import random
import time
from pathlib import Path

import requests


def load_queries(data_path: str, num_queries: int, shuffle: bool = True) -> list[dict]:
    """Load conversation entries from a JSONL file."""
    with open(data_path) as f:
        entries = [json.loads(line) for line in f]

    if shuffle:
        random.shuffle(entries)

    return entries[:num_queries]


def build_request(entry: dict, model_name: str, max_tokens: int, no_think: bool,
                   temperature: float = 0.7, repetition_penalty: float = 1.0,
                   no_fences: bool = False) -> dict:
    """Build an OpenAI-compatible chat completion request from a training entry."""
    messages = []
    for msg in entry["messages"]:
        if msg["role"] == "assistant":
            continue  # skip ground truth
        content = msg["content"]
        if msg["role"] == "system":
            if no_think:
                content = content.rstrip() + "\n/no_think"
            if no_fences:
                content = content.rstrip() + "\nDo not use markdown fences. Output raw JSON only."
        messages.append({"role": msg["role"], "content": content})

    return {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
    }


def call_vllm(base_url: str, payload: dict) -> dict:
    """Send a chat completion request to the vLLM server."""
    url = f"{base_url}/v1/chat/completions"
    resp = requests.post(url, json=payload, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Evaluate vLLM-served model using training queries")
    parser.add_argument("--data", default="Model-Pipeline/data/training/train.jsonl", help="Path to JSONL file")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to send")
    parser.add_argument("--model", default="fitsense-bf16", help="Served model name")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Max output tokens")
    parser.add_argument("--no-think", action="store_true", help="Append /no_think to system prompt")
    parser.add_argument("--no-fences", action="store_true", help="Append 'no markdown fences' instruction to system prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (1.0=off)")
    parser.add_argument("--output", default="Model-Deployment/eval_results.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    entries = load_queries(args.data, args.num_queries)
    print(f"Loaded {len(entries)} queries from {args.data}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as out_f:
        for i, entry in enumerate(entries):
            payload = build_request(entry, args.model, args.max_tokens, args.no_think,
                                    args.temperature, args.repetition_penalty,
                                    args.no_fences)
            user_msg = next(m["content"] for m in payload["messages"] if m["role"] == "user")
            print(f"\n[{i+1}/{len(entries)}] Sending query...")
            print(f"  User: {user_msg[:120]}...")

            start = time.time()
            try:
                result = call_vllm(args.base_url, payload)
                elapsed = time.time() - start
                choice = result["choices"][0]
                content = choice["message"]["content"]
                usage = result["usage"]

                print(f"  Tokens: {usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion")
                print(f"  Time: {elapsed:.1f}s")
                print(f"  Response preview: {content[:200]}...")

                record = {
                    "query_index": i,
                    "user_message": user_msg,
                    "ground_truth": next((m["content"] for m in entry["messages"] if m["role"] == "assistant"), None),
                    "model_response": content,
                    "finish_reason": choice["finish_reason"],
                    "usage": usage,
                    "elapsed_seconds": round(elapsed, 2),
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                record = {"query_index": i, "user_message": user_msg, "error": str(e)}

            out_f.write(json.dumps(record) + "\n")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
