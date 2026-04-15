from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import threading
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SYSTEM_PLAN_PROMPT = (
    "You are FitSense AI, an expert fitness coach and periodization specialist. "
    "Return ONLY valid JSON with this exact structure: "
    '{"plan_name":"...","days":[{"name":"...","day_order":1,"notes":null,'
    '"exercises":[{"exercise_name":"...","position":1,"notes":null,'
    '"sets":[{"set_number":1,"target_reps":10,"target_rir":2,"rest_seconds":60}]}]}]}'
    ". No markdown fences. No prose before or after the JSON. Always respect injuries, conditions, equipment, and experience level."
)


class PredictInstance(BaseModel):
    task: str = Field(default="text")
    system_prompt: str | None = None
    user_message: str
    max_new_tokens: int = 768


class PredictRequest(BaseModel):
    instances: list[PredictInstance]
    parameters: dict[str, Any] | None = None


app = FastAPI(title="FitSenseAI Vertex Inference")
MODEL = None
TOKENIZER = None
MODEL_READY = False
MODEL_ERROR: str | None = None
MODEL_DIR = Path("/tmp/fitsense_model")
_model_lock = threading.Lock()
_load_event = threading.Event()


def _download_gcs_dir(gcs_uri: str, target: Path) -> None:
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gcs_uri}")
    bucket_name, prefix = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix):].lstrip("/")
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(out)


def _prepare_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source = (
        os.environ.get("AIP_STORAGE_URI")
        or os.environ.get("FITSENSE_ADAPTER_GCS_URI")
        or os.environ.get("FITSENSE_STUDENT_ADAPTER_PATH")
    )
    if not source:
        raise RuntimeError("Set AIP_STORAGE_URI or FITSENSE_ADAPTER_GCS_URI or FITSENSE_STUDENT_ADAPTER_PATH")

    target = MODEL_DIR / "artifact"
    if target.exists():
        return _find_adapter_dir(target)

    if source.startswith("gs://"):
        _download_gcs_dir(source, target)
    else:
        src = Path(source)
        if src.is_dir():
            shutil.copytree(src, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    if target.is_file() and target.suffix == ".zip":
        with zipfile.ZipFile(target) as zf:
            zf.extractall(target.parent / "unzipped")
        return _find_adapter_dir(target.parent / "unzipped")
    if target.is_file() and (str(target).endswith(".tar.gz") or target.suffix == ".tar"):
        out_dir = target.parent / "untarred"
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target) as tf:
            tf.extractall(out_dir)
        return _find_adapter_dir(out_dir)

    return _find_adapter_dir(target)


def _find_adapter_dir(root: Path) -> Path:
    if (root / "adapter_config.json").exists():
        return root
    for p in root.rglob("adapter_config.json"):
        return p.parent
    raise RuntimeError(f"No adapter_config.json found under {root}")


def _base_model_from_adapter(adapter_dir: Path) -> str:
    # Use baked-in model if available (set by Dockerfile ENV)
    baked = os.environ.get("FITSENSE_BASE_MODEL_PATH")
    if baked and Path(baked).exists():
        return baked
    data = json.loads((adapter_dir / "adapter_config.json").read_text())
    return data.get("base_model_name_or_path") or data.get("base_model") or "Qwen/Qwen3-4B"


def _decode(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return text[start : start + end]
        except Exception:
            continue
    return None


def _build_chatml(system_prompt: str, user_message: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_message} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n</think>\n"
    )


def _load_model_background() -> None:
    global MODEL, TOKENIZER, MODEL_READY, MODEL_ERROR
    print("[fitsense] background loader: starting", flush=True)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_dir = _prepare_model_dir()
        print(f"[fitsense] adapter_dir={adapter_dir}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        base_model_id = _base_model_from_adapter(adapter_dir)
        print(f"[fitsense] base_model={base_model_id}", flush=True)

        kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["attn_implementation"] = "flash_attention_2"
            print("[fitsense] cuda available, using bfloat16 + flash_attention_2 + device_map=auto", flush=True)
        else:
            # bfloat16 on CPU: ~8 GB for 4B model — fits in 16 GB Cloud Run
            kwargs["torch_dtype"] = torch.bfloat16
            print("[fitsense] cpu-only, using bfloat16 (~8 GB)", flush=True)

        model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
        print("[fitsense] base model loaded", flush=True)

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print("[fitsense] peft adapter loaded", flush=True)

        model.eval()

        with _model_lock:
            MODEL = model
            TOKENIZER = tokenizer
            MODEL_READY = True
            MODEL_ERROR = None

        print("[fitsense] model ready", flush=True)

    except Exception as e:
        with _model_lock:
            MODEL_ERROR = repr(e)
            MODEL_READY = False
        print(f"[fitsense] model load FAILED: {MODEL_ERROR}", flush=True)
    finally:
        _load_event.set()


def _wait_for_model(timeout: float = 600.0) -> None:
    """Block until model is loaded or timeout is hit."""
    _load_event.wait(timeout=timeout)
    with _model_lock:
        if not MODEL_READY:
            raise RuntimeError(f"Model not ready: {MODEL_ERROR or 'still loading'}")


def _generate(system_prompt: str, user_message: str, max_new_tokens: int) -> str:
    _wait_for_model()
    import torch

    prompt = _build_chatml(system_prompt, user_message)
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = TOKENIZER.decode(gen_ids, skip_special_tokens=True)
    return _decode(decoded)


@app.on_event("startup")
def startup_event() -> None:
    # Bind port immediately, load model in background thread.
    print("[fitsense] startup: server listening, kicking off background model load", flush=True)
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ready": MODEL_READY,
        "error": MODEL_ERROR,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    print(f"[fitsense] /predict entered, instances={len(payload.instances)}", flush=True)

    _wait_for_model()

    predictions = []
    for i, inst in enumerate(payload.instances, start=1):
        print(f"[fitsense] processing instance {i} task={inst.task}", flush=True)

        system_prompt = inst.system_prompt or SYSTEM_PLAN_PROMPT
        raw_text = _generate(system_prompt, inst.user_message, inst.max_new_tokens)

        print(f"[fitsense] instance {i} raw_text_preview={raw_text[:300]!r}", flush=True)

        if inst.task == "plan_json":
            candidate = _extract_first_json_object(raw_text)
            if not candidate:
                print(f"[fitsense] instance {i} no JSON object found", flush=True)
                raise HTTPException(status_code=500, detail="Model did not return a JSON object")
            plan_json = json.loads(candidate)
            predictions.append({"plan_json": plan_json, "raw_text": raw_text})
            print(f"[fitsense] instance {i} JSON parsed successfully", flush=True)
        else:
            predictions.append({"text": raw_text, "raw_text": raw_text})
            print(f"[fitsense] instance {i} text response ready", flush=True)

    print("[fitsense] /predict completed", flush=True)
    return {"predictions": predictions}
