"""Tests for ssa.utils.handle_config dataset dispatch + error paths."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ssa.utils import handle_config as hc


def test_tb2_raises_without_env_var(monkeypatch):
    """TB2 path raises ValueError when TB2_INSTRUCTIONS_MAP is unset."""
    monkeypatch.delenv("TB2_INSTRUCTIONS_MAP", raising=False)
    with pytest.raises(ValueError, match="TB2_INSTRUCTIONS_MAP"):
        hc.get_problem_statement_for_tb2("some-task")


def test_tb2_missing_identifier_raises(tmp_path, monkeypatch):
    """TB2 path raises KeyError when identifier is not in the map."""
    m = tmp_path / "map.json"
    m.write_text(json.dumps({"known-task": "do the thing"}))
    monkeypatch.setenv("TB2_INSTRUCTIONS_MAP", str(m))

    with pytest.raises(KeyError, match="unknown-task"):
        hc.get_problem_statement_for_tb2("unknown-task")


def test_unknown_dataset_raises():
    """identifier_to_problem_statement rejects unknown dataset names."""
    with pytest.raises(ValueError, match="Unsupported dataset name"):
        hc.identifier_to_problem_statement("x", "not-a-dataset")


def test_sbv_uses_local_cache_when_available(tmp_path, monkeypatch):
    """SBV reads from the local cached HF dataset when the env var points at
    an existing path."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("HF_SBV_DATASET_OFFLINE_LOCATION", str(cache_dir))

    fake_row = {"instance_id": "django__django-1", "problem_statement": "Fix the bug"}

    class _FakeSplit:
        def filter(self, pred):
            keep = [fake_row] if pred(fake_row) else []
            return _FakeSplit._Hit(keep)

        class _Hit:
            def __init__(self, rows):
                self._rows = rows

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, i):
                return self._rows[i]

    fake_dataset = {"test": _FakeSplit()}

    with patch.object(hc, "load_from_disk", return_value=fake_dataset) as mock_load:
        result = hc.get_problem_statement_for_swebench("sbv", "django__django-1")

    assert result == "Fix the bug"
    mock_load.assert_called_once_with(str(cache_dir))
