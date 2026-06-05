"""
Docker environment extended with synchronous execution methods for SSA.

Inherits all of DockerEnvironment's async methods (start, stop, exec, upload_file,
etc.) so harbor's orchestration and verification work unchanged. Adds sync methods
(execute_bash, write_file, dir_exists, file_exists) that SSA's tools call directly
via subprocess.run() — no async/sync bridge needed.
"""

import logging
import os
import shlex
import subprocess
import tempfile
import time

from omegaconf import DictConfig, ListConfig
from strands.agent import Agent, AgentResult

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.environment_type import EnvironmentType

from ssa.environments._streaming_exec import run_with_streaming_capture

LOG = logging.getLogger(__name__)


class SSADockerEnvironment(DockerEnvironment):
    """DockerEnvironment with sync methods for SSA tool compatibility."""

    def __init__(self, *args, workdir: str = "/app", **kwargs):
        super().__init__(*args, **kwargs)
        self._workdir = workdir
        self._default_timeout: float | None = None
        self._cfg: DictConfig | None = None
        self.init_commit: str = ""
        self.retry_ctrs: dict[str, int] = {}

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.SSA_DOCKER

    # ── Sync helpers (used by SSA tools) ──────────────────────────────────

    def _docker_compose_base_cmd(self) -> list[str]:
        """The docker compose prefix shared by both async and sync paths."""
        cmd = [
            "docker", "compose",
            "-p", self.session_id.lower().replace(".", "-"),
            "--project-directory", str(self.environment_dir.resolve().absolute()),
        ]
        for path in self._docker_compose_paths:
            cmd.extend(["-f", str(path.resolve().absolute())])
        return cmd

    def _run_sync(
        self, command: list[str], timeout_sec: int | None = None
    ) -> ExecResult:
        """Run a docker compose command, streaming stdout so partial output survives a timeout."""
        full_command = self._docker_compose_base_cmd() + command
        return_code, stdout = run_with_streaming_capture(
            full_command,
            env=self._env_vars.to_env_dict(include_os_env=True),
            timeout_sec=timeout_sec,
        )
        return ExecResult(
            stdout=stdout or None,
            stderr=None,
            return_code=return_code,
        )

    # ── SSA Environment interface (sync, duck-typed) ──────────────────────
    @property
    def timeout(self) -> float | None:
        return self._default_timeout

    @property
    def workdir(self) -> str:
        return self._workdir

    def execute_bash(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 200,
        verbose: bool = False,
    ) -> dict:
        workdir = workdir or self._workdir
        start_time = time.time()
        if verbose:
            LOG.info(f"Executing in container: {command} (cwd={workdir})")

        # Wrap with in-container `timeout`: SIGTERM after {timeout}s, SIGKILL 30s later.
        exec_command = ["exec", "-T", "-e", "PYTHONUNBUFFERED=1"]
        if workdir:
            exec_command.extend(["-w", workdir])
        exec_command.append("main")
        exec_command.extend(
            ["timeout", "-k", "15s", f"{timeout}s", "bash", "-lc", command]
        )

        # Local grace window beyond the in-container timeout, in case docker compose
        # itself hangs. The in-container timeout is the primary deadline.
        result = self._run_sync(exec_command, timeout_sec=timeout + 60)

        stdout = result.stdout or ""
        exit_code = result.return_code
        status = "success" if exit_code == 0 else "error"
        # stderr is merged into stdout via Popen(stderr=STDOUT), so there is no
        # separate error stream.
        if exit_code == 0 or exit_code == 124:
            error = ""
        else:
            error = stdout

        if exit_code == 124:
            LOG.info(f"command timed out after {timeout}s; captured {len(stdout)} chars of partial output")

        if verbose:
            LOG.info(f"docker operation completed. Time taken={time.time()-start_time:.2f} sec. Suggested timeout={timeout} sec")

        return {
            "command": command,
            "status": status,
            "exit_code": exit_code,
            "output": stdout,
            "error": error,
        }

    def dir_exists(self, path: str) -> bool:
        result = self.execute_bash(f"test -d {shlex.quote(str(path))}", timeout=10)
        return result["exit_code"] == 0

    def file_exists(self, path: str) -> bool:
        result = self.execute_bash(f"test -f {shlex.quote(str(path))}", timeout=10)
        return result["exit_code"] == 0

    def write_file(self, content: str, dest_path: str) -> None:
        """Write string content to a file in the container via docker compose cp."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp", delete=False
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            cp_command = self._docker_compose_base_cmd() + [
                "cp", tmp_path, f"main:{dest_path}",
            ]
            subprocess.run(
                cp_command,
                env=self._env_vars.to_env_dict(include_os_env=True),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
        finally:
            os.unlink(tmp_path)

    def _retry_on_empty_git(self, agent: Agent, **kwargs) -> str | None:
        max_attempts = kwargs.get("max_attempts", 1)
        self.retry_ctrs["retry_on_no_change"] = self.retry_ctrs.get("retry_on_no_change", 0) + 1
        if self.retry_ctrs["retry_on_no_change"] > max_attempts:
            return None
        command = f"git -C {self.workdir} diff --name-only HEAD"
        result = self.execute_bash(command=command)
        rel_files = [f for f in result.get("output", "").splitlines() if f.strip()]
        if len(rel_files) == 0:
            return "No code-files have been edited from the current state of the workdir. Please try again!"
        return None

    def _retry_on_empty_submit(self, agent: Agent, result: AgentResult, **kwargs) -> str | None:
        max_attempts = kwargs.get("max_attempts", 1)
        self.retry_ctrs["retry_on_empty_submit"] = self.retry_ctrs.get("retry_on_empty_submit", 0) + 1
        if self.retry_ctrs["retry_on_empty_submit"] > max_attempts:
            return None

        paths = result.state.get("submit_paths", [])
        if not paths:
            return None

        # Stage all changes so new files are visible to git diff
        self.execute_bash(command=f"git -C {self.workdir} add .")

        # Get files with actual diffs against the init commit
        diff_cmd = f"git -C {self.workdir} diff --name-only {self.init_commit}"
        diff_result = self.execute_bash(command=diff_cmd)
        changed_files = set(diff_result.get("output", "").strip().splitlines())

        # Check each submitted path: it must have a non-zero diff or be a new file
        empty_paths = []
        for p in paths:
            rel_path = os.path.relpath(p, self.workdir) if os.path.isabs(p) else p
            if rel_path not in changed_files:
                empty_paths.append(p)

        if empty_paths:
            return (
                f"The following submitted paths have no changes relative to the original state: {empty_paths}. "
                f"Please ensure you have actually edited these files, or remove them from the submission and try again!"
            )
        return None

    def _retry_second_chance(self, agent: Agent, result: AgentResult, **kwargs) -> str | None:
        max_attempts = kwargs.get("max_attempts", 1)
        self.retry_ctrs["retry_second_chance"] = self.retry_ctrs.get("retry_second_chance", 0) + 1
        if self.retry_ctrs["retry_second_chance"] > max_attempts:
            return None

        if "submit" not in agent.tool_registry.registry:
            from ssa.tools import submit as submit_tool
            agent.tool_registry.process_tools([submit_tool])

        return (
            "Before you wrap this up, please take one more careful pass. "
            "Re-read the original task, walk through your changes, and consider "
            "whether every requirement is addressed and whether any edge cases "
            "deserve attention. If the solution already looks solid, that's a "
            "fine conclusion — no need to invent changes. If you do spot "
            "something worth tightening, fix it. Once you're confident the "
            "work is complete, call the submit tool to finish. Calling "
            "submit tool is the only way to properly finish the task."
        )

    def retry_feedback(self, agent: Agent, result: AgentResult) -> str | None:
        if self._cfg is None or not hasattr(self._cfg.env, "retry"):
            return None
        retry_cfg = self._cfg.env.retry
        if isinstance(retry_cfg, ListConfig):
            entries = [(rt_type, rt_cfg) for rt in retry_cfg for rt_type, rt_cfg in rt.items()]
        elif isinstance(retry_cfg, DictConfig):
            entries = list(retry_cfg.items())
        else:
            return None
        for rt_type, rt_cfg in entries:
            feedback: str | None = None
            match rt_type:
                case "retry_on_no_change":
                    feedback = self._retry_on_empty_git(agent=agent, **rt_cfg)
                case "retry_on_empty_submit":
                    feedback = self._retry_on_empty_submit(agent=agent, result=result, **rt_cfg)
                case "retry_second_chance":
                    feedback = self._retry_second_chance(agent=agent, result=result, **rt_cfg)
            if feedback:
                return feedback
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
