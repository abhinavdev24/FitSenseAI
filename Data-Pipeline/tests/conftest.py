from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


@pytest.fixture
def module_loader():
    loaded_modules: list[str] = []

    def _load(relative_path: str, module_name: str | None = None):
        path = PIPELINE_ROOT / relative_path
        if module_name is None:
            module_name = f"test_mod_{path.stem}_{uuid.uuid4().hex}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded_modules.append(module_name)
        return module

    yield _load

    for name in loaded_modules:
        sys.modules.pop(name, None)
