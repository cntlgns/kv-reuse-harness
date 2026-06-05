"""Tests for ssa.environments.environment.LocalEnvironment."""

from __future__ import annotations

from ssa.environments.environment import LocalEnvironment
from tests.conftest import make_cfg


def _make_env(workdir: str, timeout: float | None = 30) -> LocalEnvironment:
    cfg = make_cfg(env={"env_type": "local", "timeout": timeout, "local": {"workdir": workdir}})
    return LocalEnvironment(cfg, output_dir=workdir)


def test_local_env_execute_basic(tmp_path):
    """execute_bash on a successful command returns the right shape."""
    env = _make_env(str(tmp_path))
    result = env.execute_bash("echo hi")

    assert result["status"] == "success"
    assert result["exit_code"] == 0
    assert "hi" in result["output"]
    assert result["error"] == ""
    assert result["command"] == "echo hi"


def test_local_env_execute_failure(tmp_path):
    """A failing command surfaces exit_code, status=error, and a populated error."""
    env = _make_env(str(tmp_path))
    result = env.execute_bash("false")

    assert result["status"] == "error"
    assert result["exit_code"] != 0
    # Per the code path: error == output when nonzero AND not 124.
    assert result["error"] == result["output"]


def test_local_env_timeout_captures_partial(tmp_path):
    """Timeout (exit_code 124) keeps partial stdout and doesn't populate ``error``."""
    env = _make_env(str(tmp_path), timeout=1)
    # ``PYTHONUNBUFFERED=1`` is already set by the env. Using bash echo so output
    # flushes immediately regardless.
    result = env.execute_bash("echo PARTIAL; sleep 10")

    assert result["exit_code"] == 124
    assert "PARTIAL" in result["output"]
    assert result["error"] == ""
