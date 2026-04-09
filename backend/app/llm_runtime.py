from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

SYSTEM_PLAN_PROMPT = (
    "You are FitSense AI, an expert fitness coach and periodization specialist. "
    "Return ONLY valid JSON with this exact structure: "
    '{"plan_name":"...","days":[{"name":"...","day_order":1,"notes":null,'
    '"exercises":[{"exercise_name":"...","position":1,"notes":null,'
    '"sets":[{"set_number":1,"target_reps":10,"target_rir":2,"rest_seconds":60}]}]}]}'
    ". No markdown fences. No prose before or after the JSON. Always respect injuries, conditions, equipment, and experience level."
)

SYSTEM_COACH_PROMPT = (
    "You are FitSense AI, a conservative safety-aware fitness coach. "
    "Reply in plain text using 2-5 sentences. Give concrete and safe guidance. "
    "If pain, injury, or sharp symptoms are mentioned, explicitly advise the user to stop painful movements and consider a clinician if symptoms are severe or worsening."
)


@dataclass
class RuntimeInfo:
    available: bool
    provider: str
    base_model: str | None
    adapter_path: str | None
    registry_record: str | None
    reason: str | None = None
    detail: str | None = None
    registry_adapter_hint: str | None = None
    optional_dependencies_ready: bool = False
    adapter_files_ready: bool = False
    full_model_files_ready: bool = False
    artifact_uri: str | None = None
    last_load_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StudentLLMRuntime:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.model_root = self.repo_root / "Model-Pipeline"
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None
        self._loaded_adapter_path: str | None = None
        self.adapter_path: Path | None = None
        self.registry_record: Path | None = None
        self.base_model: str | None = None
        self.registry_adapter_hint: str | None = None
        self.artifact_uri: str | None = None
        self.runtime_mode: str = "adapter"
        self.cache_dir = self.model_root / "adapters" / ".downloaded"
        self.cloudrun_service_url: str | None = None
        self.cloud_predict_url: str | None = None
        self.refresh_configuration(force=True)

    def refresh_configuration(self, force: bool = False) -> None:
        old_adapter = str(self.adapter_path) if self.adapter_path else None
        old_base = self.base_model
        old_mode = self.runtime_mode
        old_cloud = self.cloud_predict_url
        self.registry_record = self._discover_registry_record()
        self.artifact_uri = self._artifact_uri_from_registry()
        self.adapter_path, self.runtime_mode = self._discover_runtime_path()
        self.base_model = (
            os.environ.get("FITSENSE_STUDENT_BASE_MODEL")
            or self._base_model_from_runtime_path(self.adapter_path, self.runtime_mode)
            or self._base_model_from_registry()
            or "unsloth/Qwen3-4B-bnb-4bit"
        )
        self.cloudrun_service_url = (os.environ.get("FITSENSE_CLOUDRUN_URL") or "").strip() or None
        self.cloud_predict_url = (self.cloudrun_service_url.rstrip("/") + "/predict") if self.cloudrun_service_url else None
        new_adapter = str(self.adapter_path) if self.adapter_path else None
        if force or new_adapter != old_adapter or self.base_model != old_base or self.runtime_mode != old_mode or self.cloud_predict_url != old_cloud:
            self._model = None
            self._tokenizer = None
            self._loaded_adapter_path = None
            self._load_error = None

    def _read_registry_data(self) -> dict[str, Any] | None:
        if not self.registry_record or not self.registry_record.exists():
            return None
        try:
            return json.loads(self.registry_record.read_text())
        except Exception:
            return None

    def _discover_registry_record(self) -> Path | None:
        explicit = os.environ.get("FITSENSE_STUDENT_REGISTRY_RECORD")
        if explicit:
            p = Path(explicit)
            return p if p.exists() else None
        reports = self.model_root / "reports"
        if not reports.exists():
            return None
        latest_pointer = reports / "latest_student_adapter.json"
        if latest_pointer.exists():
            return latest_pointer
        records = sorted(reports.glob("registry_record_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return records[0] if records else None

    def _base_model_from_registry(self) -> str | None:
        data = self._read_registry_data()
        return data.get("base_model") if data else None

    def _artifact_uri_from_registry(self) -> str | None:
        data = self._read_registry_data()
        if not data:
            return None
        return data.get("artifact_uri") or data.get("gcs_uri") or data.get("vertex_resource_name")

    def _base_model_from_runtime_path(self, path: Path | None, mode: str) -> str | None:
        if path is None:
            return None
        if mode == "adapter":
            fp = path / "adapter_config.json"
            if fp.exists():
                try:
                    data = json.loads(fp.read_text())
                    return data.get("base_model_name_or_path") or data.get("base_model")
                except Exception:
                    pass
        for fn in ["config.json", "tokenizer_config.json", "generation_config.json"]:
            fp = path / fn
            if fp.exists():
                try:
                    data = json.loads(fp.read_text())
                    for key in ["_name_or_path", "base_model_name_or_path", "base_model", "model_name", "model"]:
                        val = data.get(key)
                        if isinstance(val, str) and val:
                            return val
                except Exception:
                    pass
        return None

    def _has_module(self, name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def _optional_dependencies_ready(self) -> bool:
        return self._has_module("torch") and self._has_module("transformers") and self._has_module("peft")

    def _adapter_files_ready(self, path: Path | None) -> bool:
        if path is None:
            return False
        return (path / "adapter_config.json").exists() and (
            (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
        )

    def _looks_like_full_model_dir(self, path: Path | None) -> bool:
        if path is None or not path.exists() or not path.is_dir():
            return False
        has_config = (path / "config.json").exists()
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
        return has_config and has_weights

    def _artifact_is_archive(self, path: Path) -> bool:
        name = path.name.lower()
        return name.endswith(".zip") or name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz")

    def _extract_archive_to_dir(self, archive_path: Path, out_dir: Path) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if archive_path.name.lower().endswith(".zip"):
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(out_dir)
            else:
                with tarfile.open(archive_path) as tf:
                    tf.extractall(out_dir)
        except Exception as exc:
            self._load_error = f"Failed to extract student artifact {archive_path}: {exc}"
            return None
        candidates = [out_dir] + [p for p in out_dir.rglob("*") if p.is_dir()]
        for c in candidates:
            if self._adapter_files_ready(c) or self._looks_like_full_model_dir(c):
                return c
        self._load_error = (
            f"Extracted student artifact {archive_path}, but no adapter directory or full model directory was found inside it."
        )
        return None

    def _materialize_registry_artifact(self) -> tuple[Path | None, str | None]:
        uri = os.environ.get("FITSENSE_STUDENT_ARTIFACT") or self.artifact_uri
        if not uri:
            return None, None

        # Local file or directory
        if not re.match(r"^[a-zA-Z]+://", uri):
            src = Path(uri)
            if not src.exists():
                self._load_error = f"Configured student artifact path does not exist: {src}"
                return None, None
            if src.is_dir():
                if self._adapter_files_ready(src):
                    return src.resolve(), "adapter"
                if self._looks_like_full_model_dir(src):
                    return src.resolve(), "full-model"
                self._load_error = f"Configured student artifact directory exists but does not look like an adapter or full model: {src}"
                return None, None
            if src.is_file() and self._artifact_is_archive(src):
                found = self._extract_archive_to_dir(src, self.cache_dir / src.stem)
                if found:
                    return found.resolve(), "full-model" if self._looks_like_full_model_dir(found) else "adapter"
            self._load_error = f"Configured student artifact is not a supported archive or model directory: {src}"
            return None, None

        # gs:// artifact via gsutil if available
        if uri.startswith("gs://"):
            if not shutil.which("gsutil"):
                self._load_error = (
                    f"Registry points to {uri}, but gsutil is not installed, so the backend cannot download the student artifact automatically."
                )
                return None, None
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            local_name = uri.rstrip("/").split("/")[-1]
            local_path = self.cache_dir / local_name
            if not local_path.exists():
                try:
                    subprocess.run(["gsutil", "cp", uri, str(local_path)], check=True, capture_output=True, text=True)
                except Exception as exc:
                    self._load_error = f"Failed to download student artifact from registry URI {uri}: {exc}"
                    return None, None
            if local_path.is_dir():
                if self._adapter_files_ready(local_path):
                    return local_path.resolve(), "adapter"
                if self._looks_like_full_model_dir(local_path):
                    return local_path.resolve(), "full-model"
            if local_path.is_file() and self._artifact_is_archive(local_path):
                found = self._extract_archive_to_dir(local_path, self.cache_dir / local_path.stem)
                if found:
                    return found.resolve(), "full-model" if self._looks_like_full_model_dir(found) else "adapter"
            self._load_error = f"Downloaded registry artifact from {uri}, but it was not a usable adapter or full model package."
            return None, None

        self._load_error = f"Unsupported artifact URI scheme for student model: {uri}"
        return None, None

    def _discover_runtime_path(self) -> tuple[Path | None, str]:
        explicit = os.environ.get("FITSENSE_STUDENT_ADAPTER_PATH")
        candidates: list[tuple[Path, str]] = []
        self.registry_adapter_hint = None

        if explicit:
            candidates.append((Path(explicit), "adapter"))

        data = self._read_registry_data()
        if data and data.get("adapter_path"):
            self.registry_adapter_hint = str(data["adapter_path"])
            candidates.append((Path(data["adapter_path"]), "adapter"))

        model_dir = self.model_root / "adapters"
        if model_dir.exists():
            for p in sorted(model_dir.glob("**/adapter_config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                candidates.append((p.parent, "adapter"))
            for p in sorted(model_dir.glob("**/config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                candidates.append((p.parent, "full-model"))
            for p in sorted(model_dir.glob("**/adapter_model.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
                candidates.append((p.parent, "adapter"))
            for p in sorted(model_dir.glob("**/adapter_model.bin"), key=lambda p: p.stat().st_mtime, reverse=True):
                candidates.append((p.parent, "adapter"))

        for p, mode in candidates:
            p = p.resolve()
            if self._adapter_files_ready(p):
                return p, "adapter"
            if self._looks_like_full_model_dir(p):
                return p, "full-model"

        materialized, mode = self._materialize_registry_artifact()
        if materialized is not None:
            return materialized, mode or "adapter"
        return None, "adapter"

    def _status_info(self) -> RuntimeInfo:
        registry_path = str(self.registry_record) if self.registry_record else None
        adapter_path = str(self.adapter_path) if self.adapter_path else None
        deps_ready = self._optional_dependencies_ready()
        adapter_ready = self._adapter_files_ready(self.adapter_path)
        full_model_ready = self._looks_like_full_model_dir(self.adapter_path)
        provider = "student-full-model" if self.runtime_mode == "full-model" else "student-llm"

        if self.adapter_path is None:
            detail = "No local adapter directory or full model directory was found under Model-Pipeline/adapters/."
            if self.artifact_uri:
                detail += f" Registry artifact was referenced as {self.artifact_uri}, but it could not be materialized locally."
            elif self.registry_record and self.registry_record.exists():
                detail = (
                    "A registry record exists, but this project does not contain local adapter files or a local merged model. "
                    "The backend needs either an exported adapter directory with adapter_config.json + adapter weights, "
                    "or a full model directory with config.json + weight files."
                )
            return RuntimeInfo(
                False,
                "rule-fallback",
                self.base_model,
                adapter_path,
                registry_path,
                "Student model files are missing locally, so the backend will use rule-based logic.",
                detail,
                self.registry_adapter_hint,
                deps_ready,
                adapter_ready,
                full_model_ready,
                self.artifact_uri,
                self._load_error,
            )

        if not deps_ready:
            return RuntimeInfo(
                False,
                "rule-fallback",
                self.base_model,
                adapter_path,
                registry_path,
                "Student model files exist but optional inference dependencies are missing, so the backend will use rule-based logic.",
                "Install backend/requirements-llm.txt to enable transformers + peft inference.",
                self.registry_adapter_hint,
                deps_ready,
                adapter_ready,
                full_model_ready,
                self.artifact_uri,
                self._load_error,
            )

        if self.runtime_mode == "adapter" and not adapter_ready:
            return RuntimeInfo(
                False,
                "rule-fallback",
                self.base_model,
                adapter_path,
                registry_path,
                "Adapter directory is present but incomplete, so the backend will use rule-based logic.",
                "Expected adapter_config.json and adapter weights (adapter_model.safetensors or adapter_model.bin).",
                self.registry_adapter_hint,
                deps_ready,
                adapter_ready,
                full_model_ready,
                self.artifact_uri,
                self._load_error,
            )

        if self.runtime_mode == "full-model" and not full_model_ready:
            return RuntimeInfo(
                False,
                "rule-fallback",
                self.base_model,
                adapter_path,
                registry_path,
                "Full student model directory is present but incomplete, so the backend will use rule-based logic.",
                "Expected config.json and model weight files such as *.safetensors or pytorch_model*.bin.",
                self.registry_adapter_hint,
                deps_ready,
                adapter_ready,
                full_model_ready,
                self.artifact_uri,
                self._load_error,
            )

        return RuntimeInfo(
            True,
            provider,
            self.base_model,
            adapter_path,
            registry_path,
            self._load_error,
            "Student model files are available and the backend will try to use them first.",
            self.registry_adapter_hint,
            deps_ready,
            adapter_ready,
            full_model_ready,
            self.artifact_uri,
            self._load_error,
        )

    def _fallback_base_model_for_non_bnb(self, base_model: str | None) -> str | None:
        if not base_model:
            return None
        known = {
            "unsloth/Qwen3-4B-bnb-4bit": "Qwen/Qwen3-4B",
            "unsloth/Qwen3-4B-Base-bnb-4bit": "Qwen/Qwen3-4B-Base",
            "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit": "Qwen/Qwen3-4B-Instruct-2507",
            "unsloth/Qwen3-4B-unsloth-bnb-4bit": "Qwen/Qwen3-4B",
        }
        if base_model in known:
            return known[base_model]
        if base_model.startswith("unsloth/") and "bnb" in base_model.lower():
            name = base_model.split("/", 1)[1]
            name = name.replace("-unsloth-bnb-4bit", "")
            name = name.replace("-bnb-4bit", "")
            return f"Qwen/{name}"
        return None

    def _select_loading_strategy(self, torch_module):
        use_cuda = bool(torch_module.cuda.is_available())
        use_mps = bool(getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available())
        kwargs: dict[str, Any] = {}
        base_model = self.base_model
        notes: list[str] = []

        if use_cuda:
            kwargs["device_map"] = "auto"
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_module.float16,
                )
                notes.append("CUDA available, so 4-bit bitsandbytes loading is enabled.")
            except Exception as exc:
                notes.append(f"CUDA available but bitsandbytes quantization could not be enabled: {exc}")
        else:
            fallback = self._fallback_base_model_for_non_bnb(base_model)
            if fallback and fallback != base_model:
                notes.append(
                    f"Non-CUDA runtime detected on {platform.system()}; switching base model from {base_model} to {fallback} because the quantized bitsandbytes variant is not usable here."
                )
                base_model = fallback
            kwargs["torch_dtype"] = torch_module.float16 if use_mps else torch_module.float32
            notes.append("Using non-quantized loading because CUDA is unavailable.")

        return base_model, kwargs, notes

    def info(self) -> RuntimeInfo:
        self.refresh_configuration()
        if self._can_use_cloud():
            info = self._status_info()
            info.available = True
            info.provider = "cloud-run"
            info.reason = None
            info.detail = f"Cloud Run inference via {self.cloud_predict_url}"
            return info
        info = self._status_info()
        if self._load_error:
            info.last_load_error = self._load_error
            if self._model is None:
                info.reason = "Student model is configured but the last load or generation attempt failed, so the backend may fall back to rules."
                info.detail = self._load_error
            elif info.detail:
                info.detail = f"{info.detail} {self._load_error}"
            else:
                info.detail = self._load_error
        return info

    def _ensure_loaded(self) -> bool:
        self.refresh_configuration()
        if self._model is not None and self._tokenizer is not None and self._loaded_adapter_path == str(self.adapter_path):
            return True
        if self.adapter_path is None:
            return False
        if self.runtime_mode == "adapter" and not self._adapter_files_ready(self.adapter_path):
            return False
        if self.runtime_mode == "full-model" and not self._looks_like_full_model_dir(self.adapter_path):
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            PeftModel = None
            if self.runtime_mode == "adapter":
                from peft import PeftModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            self._load_error = f"Optional LLM dependencies missing: {exc}"
            return False
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
            load_base_model, kwargs, notes = self._select_loading_strategy(torch)
            if self.runtime_mode == "full-model":
                model = AutoModelForCausalLM.from_pretrained(str(self.adapter_path), **kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(load_base_model or self.base_model, **kwargs)
                model = PeftModel.from_pretrained(model, str(self.adapter_path))
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                model = model.to("mps")
            model.eval()
            self._model = model
            self._tokenizer = tokenizer
            self._loaded_adapter_path = str(self.adapter_path)
            self._load_error = " ".join(notes) if notes else None
            print(f"✅ Student LLM loaded from {self.adapter_path} (mode={self.runtime_mode})")
            if notes:
                print(f"[student-load] {' '.join(notes)}")
            return True
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            print(f"⚠️ Student LLM load failed: {exc}")
            return False

    def _build_chatml(self, system_prompt: str, user_message: str) -> str:
        return (
            "<|im_start|>system\n"
            f"{system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_message} /no_think<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n</think>\n"
        )

    def _decode(self, text: str) -> str:
        text = text.strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return text

    def _extract_first_json_object(self, text: str) -> str | None:
        if not text:
            return None

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            start = match.start()
            try:
                obj, end = decoder.raw_decode(text[start:])
                if isinstance(obj, dict):
                    return text[start:start + end]
            except Exception:
                continue
        return None

    def _repair_common_json_issues(self, text: str) -> str:
        repaired = text.strip()
        repaired = re.sub(r"^```json\s*", "", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"^```\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        return repaired

    def _can_use_cloud(self) -> bool:
        return bool(self.cloud_predict_url and importlib.util.find_spec("requests"))

    def _debug_enabled(self) -> bool:
        return (os.environ.get("FITSENSE_DEBUG_VERTEX") or "1").strip().lower() not in {"0", "false", "no", "off"}

    def _debug(self, message: str) -> None:
        if self._debug_enabled():
            print(f"[vertex-debug] {message}")

    def _call_openrouter(self, *, system_prompt: str, user_message: str, max_new_tokens: int) -> str | None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return None
        import requests
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-8b:free"),
                      "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                      "max_tokens": max_new_tokens, "temperature": 0},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            print(f"⚠️ OpenRouter call failed: {exc}")
            return None

    def _call_cloud(self, *, task: str, system_prompt: str, user_message: str, max_new_tokens: int) -> dict[str, Any] | None:
        if not self._can_use_cloud():
            return None
        import requests, time
        # Truncate long user messages to keep input tokens reasonable
        if len(user_message) > 1500:
            user_message = user_message[:1500] + "\n[profile truncated for inference]"
        payload = {"instances": [{"task": task, "system_prompt": system_prompt, "user_message": user_message, "max_new_tokens": max_new_tokens}]}
        self._debug(f"Starting Cloud Run call task={task} target={self.cloud_predict_url} chars={len(user_message)}")
        start = time.perf_counter()
        try:
            resp = requests.post(self.cloud_predict_url, headers={"Content-Type": "application/json"}, json=payload, timeout=600)
            elapsed = time.perf_counter() - start
            self._debug(f"Cloud HTTP response status={resp.status_code} elapsed={elapsed:.2f}s")
            resp.raise_for_status()
            body = resp.json()
            preds = body.get("predictions") or []
            self._debug(f"Cloud response prediction_count={len(preds)} top_level_keys={list(body.keys())}")
            if not preds:
                return None
            pred = preds[0]
            if isinstance(pred, dict):
                self._debug(f"Prediction keys={sorted(pred.keys())}")
                return pred
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._load_error = f"Cloud inference call failed: {exc}"
            print(f"⚠️ Cloud inference call failed after {elapsed:.2f}s: {exc}")
        return None

    def generate_text(self, *, system_prompt: str, user_message: str, max_new_tokens: int = 768) -> str | None:
        text = self._call_openrouter(system_prompt=system_prompt, user_message=user_message, max_new_tokens=max_new_tokens)
        if text:
            return text
        cloud_result = self._call_cloud(task="text", system_prompt=system_prompt, user_message=user_message, max_new_tokens=max_new_tokens)
        if cloud_result is not None:
            text = cloud_result.get("text") or cloud_result.get("raw_text")
            if isinstance(text, str) and text.strip():
                print(f"[llm-output] text:\n{text.strip()}")
                return text.strip()
        if not self._ensure_loaded():
            return None
        try:
            import torch
            prompt = self._build_chatml(system_prompt, user_message)
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            decoded = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            return self._decode(decoded)
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            print(f"⚠️ Student LLM generation failed: {exc}")
            return None

    def generate_plan_json(self, *, user_message: str) -> dict[str, Any] | None:
        self._debug("generate_plan_json started.")
        or_text = self._call_openrouter(system_prompt=SYSTEM_PLAN_PROMPT, user_message=user_message, max_new_tokens=2500)
        if or_text:
            candidate = self._extract_first_json_object(or_text)
            if candidate:
                try:
                    return json.loads(self._repair_common_json_issues(candidate))
                except Exception:
                    pass

        if self._can_use_cloud():
            for attempt in range(1, 4):  # up to 3 attempts
                self._debug(f"Cloud plan_json attempt {attempt}/3")
                cloud_result = self._call_cloud(task="plan_json", system_prompt=SYSTEM_PLAN_PROMPT, user_message=user_message, max_new_tokens=2500)
                if cloud_result is None:
                    self._debug("Cloud call failed (network/HTTP error) — not retrying.")
                    break
                plan_json = cloud_result.get("plan_json")
                if isinstance(plan_json, dict):
                    self._debug(f"Cloud returned structured plan_json keys={sorted(plan_json.keys())}")
                    print(f"[llm-output] plan_json:\n{json.dumps(plan_json, indent=2)}")
                    return plan_json
                maybe_text = cloud_result.get("text") or cloud_result.get("raw_text")
                if isinstance(maybe_text, str):
                    print(f"[llm-output] raw text (attempt {attempt}):\n{maybe_text}")
                    self._debug(f"Cloud returned text length={len(maybe_text)}; attempting JSON extraction.")
                    decoder = json.JSONDecoder()
                    for match in re.finditer(r"\{", maybe_text):
                        try:
                            obj, _ = decoder.raw_decode(maybe_text[match.start():])
                            if isinstance(obj, dict) and "plan_name" in obj:
                                self._debug("Extracted plan_json from text response.")
                                return obj
                        except Exception:
                            continue
                    self._debug(f"Attempt {attempt}: JSON truncated or missing plan_name — retrying.")
            self._load_error = "Cloud responded but could not produce valid plan JSON after 3 attempts."
            self._debug("All cloud attempts exhausted — not falling back to rules.")
            return None

        self._debug("No cloud configured — falling back to local generate_text.")
        text = self.generate_text(system_prompt=SYSTEM_PLAN_PROMPT, user_message=user_message, max_new_tokens=2500)
        if not text:
            return None

        candidate = self._extract_first_json_object(text)
        if not candidate:
            self._load_error = "Student model returned text but not a valid JSON object."
            print("\n=== RAW PLAN OUTPUT ===\n", text)
            return None

        candidate = self._repair_common_json_issues(candidate)

        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                self._load_error = "Student model returned JSON, but it was not an object."
                return None
            return data
        except Exception as exc:
            self._load_error = f"Student model returned malformed JSON: {exc}"
            return None

    def generate_coach_text(self, *, user_message: str) -> str | None:
        cloud_result = self._call_cloud(task="coach_text", system_prompt=SYSTEM_COACH_PROMPT, user_message=user_message, max_new_tokens=240)
        if cloud_result is not None:
            text = cloud_result.get("text") or cloud_result.get("raw_text")
            if isinstance(text, str) and text.strip():
                print(f"[llm-output] coach text:\n{text.strip()}")
                return text.strip()
        text = self.generate_text(system_prompt=SYSTEM_COACH_PROMPT, user_message=user_message, max_new_tokens=240)
        if text is None:
            return None
        cleaned = text.strip()
        if not cleaned:
            self._load_error = "Student model returned an empty coaching reply."
            return None
        return cleaned


@lru_cache(maxsize=1)
def get_runtime() -> StudentLLMRuntime:
    return StudentLLMRuntime()
