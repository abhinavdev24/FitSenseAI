"""Tests for Model-Pipeline/scripts/load_data.py.

Covers validate_schema() and compute_stats() — both are pure functions that
require only the `datasets` package and no GPU.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from datasets import Dataset, DatasetDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    roles: list[str] | None = None,
    contents: list[Any] | None = None,
    metadata: Any = None,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Build a minimal dataset row with configurable messages and metadata."""
    if roles is None:
        roles = ["system", "user", "assistant"]
    if contents is None:
        contents = ["system prompt", "user message", "<think>\nreasoning\n</think>\njson"]
    if metadata is None:
        metadata = {"provider": "groq"}

    row: dict[str, Any] = {
        "messages": [
            {"role": r, "content": c} for r, c in zip(roles, contents)
        ]
    }
    if include_metadata:
        row["metadata"] = metadata
    return row


def _silent_logger() -> logging.Logger:
    """Return a logger that discards all output (keeps test stdout clean)."""
    logger = logging.getLogger(f"test_{id(object())}")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Tests for validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_row_returns_true(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")
        row = _make_row()
        assert mod.validate_schema(row, 0, _silent_logger()) is True

    def test_missing_messages_returns_false(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")
        row = {"metadata": {}}  # no 'messages' key
        assert mod.validate_schema(row, 0, _silent_logger()) is False

    def test_wrong_message_count_returns_false(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")
        row = _make_row(
            roles=["system", "user"],
            contents=["sys", "usr"],
        )
        assert mod.validate_schema(row, 0, _silent_logger()) is False

    def test_wrong_role_returns_false(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")
        # First role should be "system", not "human"
        row = _make_row(roles=["human", "user", "assistant"])
        assert mod.validate_schema(row, 0, _silent_logger()) is False

    def test_non_string_content_returns_false(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")
        row = _make_row(contents=["sys", 42, "assistant reply"])
        assert mod.validate_schema(row, 0, _silent_logger()) is False

    def test_missing_metadata_returns_false(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")
        row = _make_row(include_metadata=False)
        assert mod.validate_schema(row, 0, _silent_logger()) is False


# ---------------------------------------------------------------------------
# Tests for compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def _make_dataset_dict(
        self,
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]] | None = None,
    ) -> DatasetDict:
        if val_rows is None:
            val_rows = train_rows[:1]
        return DatasetDict({
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
        })

    def test_thinking_ratio_is_correct(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")

        thinking_row = _make_row(
            contents=["sys", "usr", "<think>\nreason\n</think>\njson"]
        )
        plain_row = _make_row(contents=["sys", "usr", "json_only"])
        dataset_dict = self._make_dataset_dict([thinking_row, plain_row])

        stats = mod.compute_stats(dataset_dict, _silent_logger())

        train_stats = stats["train"]
        assert train_stats["row_count"] == 2
        assert train_stats["thinking_count"] == 1
        assert train_stats["non_thinking_count"] == 1
        assert train_stats["thinking_ratio"] == 0.5

    def test_provider_breakdown_is_counted(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")

        row_groq = _make_row(metadata={"provider": "groq"})
        row_openai = _make_row(metadata={"provider": "openai"})
        row_groq2 = _make_row(metadata={"provider": "groq"})
        dataset_dict = self._make_dataset_dict([row_groq, row_openai, row_groq2])

        stats = mod.compute_stats(dataset_dict, _silent_logger())

        breakdown = stats["train"]["provider_breakdown"]
        assert breakdown["groq"] == 2
        assert breakdown["openai"] == 1

    def test_approx_tokens_is_positive(self, module_loader: Any) -> None:
        mod = module_loader("scripts/load_data.py")
        dataset_dict = self._make_dataset_dict([_make_row()])
        stats = mod.compute_stats(dataset_dict, _silent_logger())
        assert stats["train"]["approx_tokens"] > 0
