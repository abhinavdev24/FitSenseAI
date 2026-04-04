from __future__ import annotations

import json


class _Logger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, msg: str, arg: str) -> None:
        self.messages.append((msg, arg))


def test_main_writes_bootstrap_report(tmp_path, module_loader, monkeypatch):
    module = module_loader("bootstrap.py")

    report_dir = tmp_path / "reports"
    params = {
        "project": {"name": "FitSenseAI"},
        "reproducibility": {"seed": 7, "hash_seed": "hash-7"},
        "logging": {
            "level": "INFO",
            "file_name": "pipeline.log",
            "format": "%(message)s",
        },
        "paths": {
            "reports_dir": str(report_dir),
            "logs_dir": str(tmp_path / "logs"),
        },
    }

    fake_logger = _Logger()
    seed_calls: list[tuple[int, str]] = []

    monkeypatch.setattr(module, "load_params", lambda: params)
    monkeypatch.setattr(
        module,
        "apply_global_seed",
        lambda seed, hash_seed: seed_calls.append((seed, hash_seed)),
    )
    monkeypatch.setattr(module, "setup_logger", lambda **_: fake_logger)

    module.main()

    report_path = report_dir / "phase1_bootstrap.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["project"] == "FitSenseAI"
    assert payload["seed"] == 7
    assert payload["hash_seed"] == "hash-7"
    assert payload["status"] == "ok"
    assert payload["phase"] == "phase1_scaffold"
    assert seed_calls == [(7, "hash-7")]
    assert fake_logger.messages
