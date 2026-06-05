"""Shared fixtures and helpers for the SSA test suite."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest
from omegaconf import DictConfig, OmegaConf

from ssa.environments.environment import Environment


def make_cfg(**overrides: Any) -> DictConfig:
    """Build a minimal hydra-style DictConfig, deep-merged with overrides."""
    base = {
        "env": {
            "env_type": "local",
            "timeout": 30,
            "local": {"workdir": "/tmp"},
            "docker": {},
            "host_mount": "/tmp",
        },
        "agent": {
            "agent_id": "test-agent",
            "model": "anthropic/claude-sonnet-4-5",
            "invoker": "bedrock",
            "invoker_params": {},
            "tools": {},
            "prompt_tag": "default",
        },
        "dataset": {"name": "sbv", "identifier": "test__repo-1"},
        "aws": {"region": "us-east-1"},
        "max_llm_iterations": 100,
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


class FakeEnvironment(Environment):
    """In-memory Environment implementation for tool tests.

    ``execute_bash`` returns a canned response dict from ``_responses`` queue,
    or a configurable default. Tracks all calls in ``calls``.
    """

    def __init__(
        self,
        workdir: str = "/work",
        timeout: float | None = 30,
        default_response: dict | None = None,
        files: dict[str, str] | None = None,
        dirs: set[str] | None = None,
    ):
        self._workdir = workdir
        self._timeout = timeout
        self._default_response = default_response or {
            "command": "",
            "status": "success",
            "exit_code": 0,
            "output": "",
            "error": "",
        }
        self._files: dict[str, str] = dict(files or {})
        self._dirs: set[str] = set(dirs or [])
        self._responses: list[dict] = []
        self.calls: list[dict] = []

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def timeout(self) -> float | None:
        return self._timeout

    def queue_response(self, **fields: Any) -> None:
        """Queue one canned response. Missing fields fall through to defaults."""
        resp = dict(self._default_response)
        resp.update(fields)
        self._responses.append(resp)

    def execute_bash(
        self,
        command: str,
        workdir: str | None = None,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> dict:
        self.calls.append(
            {"command": command, "workdir": workdir, "timeout": timeout, "verbose": verbose}
        )
        if self._responses:
            resp = self._responses.pop(0)
        else:
            resp = dict(self._default_response)
        resp = {**resp, "command": command}
        return resp

    def dir_exists(self, path: str) -> bool:
        return path in self._dirs

    def file_exists(self, path: str) -> bool:
        return path in self._files

    def write_file(self, content: str, dest_path: str) -> None:
        self._files[dest_path] = content

    def retry_feedback(self, *args, **kwargs):
        return None

    def collect_submission(self, *args, **kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def fake_env() -> FakeEnvironment:
    """A default FakeEnvironment with a /work workdir."""
    return FakeEnvironment()


@pytest.fixture
def cfg() -> DictConfig:
    """Minimal DictConfig usable across tests."""
    return make_cfg()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Initialize a real git repo in tmp_path with one committed file.

    The fixture sets committer identity via env vars to avoid touching
    user git config.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available")

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "hello.py").write_text("print('hi')\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=env, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, env=env, check=True
    )
    return tmp_path


class ShellExecutingEnvironment(FakeEnvironment):
    """Environment that actually executes bash commands in a fixed workdir.

    Used by git_ops tests so we can run real git commands inside a tmp_path
    repo without depending on DockerEnvironment or LocalEnvironment.
    """

    def __init__(self, workdir: str, init_commit: str = ""):
        super().__init__(workdir=workdir)
        self.init_commit = init_commit
        # Track a cfg so get_git_patch's env._cfg access works if needed
        self._cfg = None

    def execute_bash(
        self,
        command: str,
        workdir: str | None = None,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> dict:
        cwd = workdir or self._workdir
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout or 30,
        )
        output = (result.stdout or "") + (result.stderr or "")
        exit_code = result.returncode
        status = "success" if exit_code == 0 else "error"
        return {
            "command": command,
            "status": status,
            "exit_code": exit_code,
            "output": output,
            "error": "" if exit_code == 0 else output,
        }


@pytest.fixture
def shell_env_factory() -> Callable[[str, str], ShellExecutingEnvironment]:
    """Factory returning an env that executes bash in a real workdir."""

    def _make(workdir: str, init_commit: str = "") -> ShellExecutingEnvironment:
        return ShellExecutingEnvironment(workdir=workdir, init_commit=init_commit)

    return _make
