from __future__ import annotations

import sys
import types


def test_find_rate_limit_calls_openrouter_endpoint(module_loader, monkeypatch):
    calls: dict[str, object] = {}

    fake_requests = types.ModuleType("requests")

    class _Resp:
        def json(self):
            return {"ok": True}

    def _fake_get(url, headers):
        calls["url"] = url
        calls["headers"] = headers
        return _Resp()

    fake_requests.get = _fake_get

    fake_dotenv = types.ModuleType("dotenv")

    def _fake_load_dotenv(path):
        calls["dotenv_path"] = path

    fake_dotenv.load_dotenv = _fake_load_dotenv

    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    printed: list[tuple[tuple, dict]] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append((a, k)))

    module_loader("find_rate_limit.py")

    assert calls["dotenv_path"] == ".env.local"
    assert calls["url"] == "https://openrouter.ai/api/v1/key"
    assert calls["headers"] == {"Authorization": "Bearer test-key"}
    assert printed
