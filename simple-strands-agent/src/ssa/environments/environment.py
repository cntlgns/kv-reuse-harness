import os
from omegaconf import DictConfig, ListConfig
from strands.agent import Agent, AgentResult
from typing import Dict
from .docker_utils import DockerConnector, parse_docker_config
from ._streaming_exec import run_with_streaming_capture
from ..utils.git_ops import put_git_marker, extract_git_changes, get_git_patch
from ..utils.tb2_utils import run_tb2_eval

from abc import ABC, abstractmethod
from typing import Self
import logging

LOG = logging.getLogger(__name__)


class Environment(ABC):
    @property
    def timeout(self) -> float | None:
        """Default (optional) timeout of the environment shell executions"""
        pass
    @abstractmethod
    def execute_bash(self, command: str, workdir: str, timeout: float, verbose: bool) -> Dict:
        """Execute bash command and return result dict with status, exit_code, output, error."""
        pass

    @abstractmethod
    def dir_exists(self, path: str) -> bool:
        """Check if directory exists."""
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        pass

    @abstractmethod
    def workdir(self) -> str:
        pass

    @abstractmethod
    def write_file(self, content: str, dest_path: str) -> None:
        """Write string content to a file in the environment."""
        pass

    @abstractmethod
    def retry_feedback(self, *args, **kwargs) -> str | None:
        pass

    @abstractmethod
    def collect_submission(self, *args, **kwargs) -> str | None:
        pass

    @abstractmethod
    def __enter__(self) -> Self:
        """Context manager entry - acquire resources."""
        pass

    @abstractmethod
    def __exit__(self, *_):
        """Context manager exit - cleanup resources."""
        pass


class DockerEnvironment(Environment):
    def __init__(self, cfg: DictConfig, output_dir: str):
        self._cfg = cfg
        self.docker_config = parse_docker_config(cfg)
        self.host_mount = cfg.env.host_mount
        self.docker_connector = None
        self._default_timeout: int = cfg.env.timeout if cfg.env.timeout and float(cfg.env.timeout) > 0 else None
        self._workdir = self.docker_config.workdir
        self.output_dir = output_dir

        self.init_commit = ""
        self.retry_ctrs: Dict[str, int] = {}

    @property
    def workdir(self) -> str:
        return self._workdir
    
    @property
    def timeout(self) -> float | None:
        return self._default_timeout

    def execute_bash(self, command: str, workdir: str | None = None, timeout: float | None = None, verbose: bool = False):
        if not self.docker_connector:
            raise RuntimeError("DockerEnvironment not initialized. Use 'with' statement.")

        timeout = self.timeout if timeout is None else timeout
        workdir = workdir or self.workdir
        exit_code, output = self.docker_connector.exec(command, cwd=workdir, verbose=verbose, timeout_sec=timeout)

        status = "success" if exit_code == 0 else "error"
        # docker exec merges stderr into stdout, so there is no separate error
        # stream. For timeouts (124) the partial output is already labelled by
        # the caller
        if exit_code == 0 or exit_code == 124:
            error = ""
        else:
            error = output

        if exit_code == 124:
            LOG.info(
                f"command timed out after {timeout}s; captured {len(output or '')} chars of partial output"
            )

        return {
            "command": command,
            "status": status,
            "exit_code": exit_code,
            "output": output,
            "error": error
        }

    def dir_exists(self, path: str, workdir: str | None = None) -> bool:
        if not self.docker_connector:
            raise RuntimeError("DockerEnvironment not initialized. Use 'with' statement.")
        workdir = workdir or self.workdir
        exit_code, _ = self.docker_connector.exec(f"test -d {path}", cwd=workdir, verbose=False)
        return exit_code == 0

    def file_exists(self, path: str, workdir: str | None = None) -> bool:
        if not self.docker_connector:
            raise RuntimeError("DockerEnvironment not initialized. Use 'with' statement.")
        workdir = workdir or self.workdir
        exit_code, _ = self.docker_connector.exec(f"test -f {path}", cwd=workdir, verbose=False)
        return exit_code == 0

    def write_file(self, content: str, dest_path: str) -> None:
        if not self.docker_connector:
            raise RuntimeError("DockerEnvironment not initialized. Use 'with' statement.")
        self.docker_connector.write_file(content.encode(), dest_path)

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

        # Extract git changes for SWE-bench datasets
        paths = []
        if self._cfg.dataset.name in ("sbv", "sbpro"):
            if result and result.state.get("submit_paths"):
                paths = result.state.get("submit_paths") 
        patch = get_git_patch(self, paths=paths, cleanup=True)

        base_feedback = (
            "Before you wrap this up, please take one more careful pass. "
            "Re-read the original task, walk through your changes, and consider "
            "whether every requirement is addressed and whether any edge cases "
            "deserve attention. If the solution already looks solid, that's a "
            "fine conclusion — no need to invent changes. If you do spot "
            "something worth tightening, fix it. Once you're confident the "
            "work is complete, call the submit tool to finish. Calling "
            "submit tool is the only way to properly finish the task."
        )
        feedback = base_feedback + (
            f"\nHere is the consolidated code-change from your current work:\n{patch}" if len(patch)>0 else ""
        )
        return feedback

    def retry_feedback(self, agent: Agent, result: AgentResult) -> str | None:
        if not hasattr(self._cfg.env, "retry"):
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

    def collect_submission(self, agent: Agent, result: AgentResult | None) -> None:
        # Extract git changes for SWE-bench datasets
        if self._cfg.dataset.name in ("sbv", "sbpro"):
            if result and result.state.get("submit_paths"):
                paths = result.state.get("submit_paths") 
            else:
                paths = []
            extract_git_changes(self, self.output_dir, paths=paths)
        return None

    def __enter__(self):
        """Initialize docker container and acquire resources."""
        LOG.info("Initializing DockerEnvironment...")
        self.docker_connector = DockerConnector(self.docker_config, self.host_mount)
        self.docker_connector.init_docker()

        # For sbv and sbpro, put a dummy commit to be used later to extract changes created in env
        if self._cfg.dataset.name in ("sbv", "sbpro"):
            self.init_commit = put_git_marker(self)

        LOG.info("DockerEnvironment initialized successfully")
        return self

    def __exit__(self, *_):
        """Cleanup docker resources on exit."""
        LOG.info("Cleaning up DockerEnvironment...")


        # Run TB2 evaluation before docker cleanup
        if self._cfg.dataset.name == "tb2":
            LOG.info("Running TB2 evaluation...")
            try:
                run_tb2_eval(self._cfg, self, self.output_dir)
            except Exception as e:
                LOG.error(f"TB2 evaluation failed: {e}")
                LOG.exception(e)

        # Cleanup docker container
        if self.docker_connector:
            self.docker_connector.cleanup()
        return False  # Don't suppress exceptions


class LocalEnvironment(Environment):
    def __init__(self, cfg: DictConfig, output_dir: str):
        self._workdir: str = cfg.env.local.workdir
        self._default_timeout: float = cfg.env.timeout if cfg.env.timeout and float(cfg.env.timeout) > 0 else None
        self.output_dir = output_dir

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def timeout(self) -> float | None:
        return self._default_timeout

    def execute_bash(self, command, workdir: str | None = None, timeout: float | None = None, verbose: bool = False):
        timeout = self.timeout if timeout is None else timeout
        workdir = workdir or self.workdir
        if verbose:
            LOG.info(f"Executing locally: {command} in workdir: {workdir}")

        # Wrap with `timeout -k 30s` so the user process gets SIGTERM (→ 30s
        # flush window → SIGKILL), giving it a chance to drain stdout before
        # death. PYTHONUNBUFFERED=1 defeats Python's pipe block-buffering.
        if timeout:
            argv = ["timeout", "-k", "30s", f"{timeout}s", "bash", "-c", command]
            outer_timeout = float(timeout) + 60
        else:
            argv = ["bash", "-c", command]
            outer_timeout = None
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        try:
            exit_code, output = run_with_streaming_capture(
                argv,
                cwd=workdir,
                env=env,
                timeout_sec=outer_timeout,
            )
        except Exception as e:
            LOG.error(f"Failed to execute command locally: {e}")
            return {
                "command": command,
                "status": "error",
                "exit_code": 1,
                "output": str(e),
                "error": str(e),
            }

        status = "success" if exit_code == 0 else "error"
        if exit_code == 0 or exit_code == 124:
            error = ""
        else:
            error = output

        if exit_code == 124:
            LOG.info(
                f"command timed out after {timeout}s; captured {len(output)} chars of partial output"
            )

        return {
            "command": command,
            "status": status,
            "exit_code": exit_code,
            "output": output,
            "error": error,
        }

    def dir_exists(self, path: str) -> bool:
        import os
        return os.path.isdir(path)

    def file_exists(self, path: str) -> bool:
        import os
        return os.path.isfile(path)

    def write_file(self, content: str, dest_path: str) -> None:
        with open(dest_path, 'w') as f:
            f.write(content)

    def retry_feedback(self, *args, **kwargs) -> str | None:
        return None

    def collect_submission(self, *args, **kwargs) -> None:
        return None

    def __enter__(self):
        """Local environment requires no initialization."""
        LOG.info(f"Entering LocalEnvironment with workdir={self.workdir}")
        return self

    def __exit__(self, *_):
        """No cleanup needed for local environment."""
        LOG.info("Exiting LocalEnvironment (no cleanup needed)")
        return False


def create_environment(cfg: DictConfig, output_dir: str) -> Environment:
    """
    Factory function to create the appropriate environment based on config.

    Args:
        cfg: Configuration object with an 'env_type' field

    Returns:
        Environment instance (DockerEnvironment or LocalEnvironment)

    Example:
        with create_environment(cfg) as env:
            result = env.execute_bash("ls -la", workdir="/tmp", timeout=30, verbose=True)
    """
    env_type = cfg.env.env_type.lower()

    if env_type == 'docker':
        LOG.info(f"Creating DockerEnvironment with image={cfg.env.docker.base_image}")
        return DockerEnvironment(cfg, output_dir)
    elif env_type == 'local':
        workdir = cfg.env.local.workdir
        LOG.info(f"Creating LocalEnvironment with workdir={workdir}")
        return LocalEnvironment(cfg, output_dir)
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Must be 'docker' or 'local'.")
