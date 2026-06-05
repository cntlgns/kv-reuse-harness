import asyncio
import contextvars
import logging
import os
import threading
import uuid

from omegaconf import OmegaConf
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from ssa.utils.harbor_plugin.ssa_docker import SSADockerEnvironment
# Contextvar that tags every thread belonging to a trial.
# strands' run_async() and asyncio.to_thread() both copy contextvars,
# so child threads inherit the value set in _run_ssa().
_TRIAL_ID: contextvars.ContextVar[str] = contextvars.ContextVar("_ssa_native_trial_id", default="")

# Disable aiohttp transport in litellm to avoid event-loop-bound session issues
# when running multiple trials in parallel (each with its own asyncio.run() loop).
os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "True")


class AgentStoppedError(Exception):
    """Raised inside the worker thread to abort the strands agent loop."""
    pass


class _OtelShutdownFilter(logging.Filter):
    """Suppress noisy OpenTelemetry context-detach errors during agent shutdown.

    When AgentStoppedError unwinds through strands' generator-based event loop,
    OTel tries to detach span context tokens that were created in the to_thread
    context — the mismatch produces harmless but spammy ValueError tracebacks.
    """

    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self._stop_event = stop_event

    def filter(self, record: logging.LogRecord) -> bool:
        if self._stop_event.is_set() and "detach context" in record.getMessage().lower():
            return False
        return True


class _TrialFilter(logging.Filter):
    """Only pass log records whose contextvar matches this filter's trial_id."""

    def __init__(self, trial_id: str):
        super().__init__()
        self._trial_id = trial_id

    def filter(self, record: logging.LogRecord) -> bool:
        return _TRIAL_ID.get("") == self._trial_id


LOG = logging.getLogger(__name__)

_CALLBACK_LOG = logging.getLogger(f"{__name__}.callback")


class _LoggingCallbackHandler:
    """Like PrintingCallbackHandler but buffers stream chunks and logs complete
    blocks through the logging framework (thread-safe, trial-filter compatible)."""

    def __init__(self, stop_event: threading.Event | None = None) -> None:
        self._buf: list[str] = []
        self._reasoning_buf: list[str] = []
        self.tool_count = 0
        self._stop_event = stop_event

    def _flush_reasoning(self) -> None:
        if self._reasoning_buf:
            _CALLBACK_LOG.info("[thinking] %s", "".join(self._reasoning_buf))
            self._reasoning_buf.clear()

    def _flush_text(self) -> None:
        if self._buf:
            _CALLBACK_LOG.info("[assistant] %s", "".join(self._buf))
            self._buf.clear()

    def __call__(self, **kwargs: Any) -> None:
        if self._stop_event and self._stop_event.is_set():
            raise AgentStoppedError("Agent cancelled by harbor timeout")

        reasoning = kwargs.get("reasoningText")
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        tool_use = kwargs.get("event", {}).get("contentBlockStart", {}).get("start", {}).get("toolUse")

        if reasoning:
            self._reasoning_buf.append(reasoning)

        if data:
            # Reasoning is done once data starts streaming
            self._flush_reasoning()
            self._buf.append(data)

        if tool_use:
            # New content block — flush any pending text
            self._flush_reasoning()
            self._flush_text()
            self.tool_count += 1
            _CALLBACK_LOG.info("[tool #%d] %s", self.tool_count, tool_use["name"])

        if complete:
            self._flush_reasoning()
            self._flush_text()


class SSANative(BaseAgent):
    """SSA agent that runs LLM calls on the host, shell commands in the container."""

    def __init__(
        self,
        logs_dir: Path,
        config_name: str,
        model_name: str | None = None,
        hydra_overrides: list[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(logs_dir, model_name=model_name, *args, **kwargs)
        self._config_name = config_name
        self._hydra_overrides = hydra_overrides or []
        self._cfg = None
        self._stop_event = threading.Event()
        self._resolved_overrides: list[str] = []

    @staticmethod
    def name() -> str:
        return "ssa-native"

    def version(self) -> str | None:
        return None

    def _load_ssa_config(self, workdir: str):
        """Load SSA Hydra config via the Compose API (no @hydra.main needed)."""
        import ssa
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        ssa_configs_dir = str(Path(ssa.__file__).parent / "configs")

        overrides = list(self._hydra_overrides)
        # Ensure env_type=local and workdir are set
        if not any(o.startswith("env.env_type=") for o in overrides):
            overrides.append("env.env_type=local")
        if not any(o.startswith("env.local.workdir=") for o in overrides):
            overrides.append(f"env.local.workdir={workdir}")

        # Override model if harbor provides one
        if self.model_name and not any(o.startswith("agent.model=") for o in overrides):
            overrides.append(f"agent.model={self.model_name}")

        # Point hydra output to our logs dir
        overrides.append(f"hydra.run.dir={self.logs_dir / 'ssa-output'}")

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=ssa_configs_dir, version_base=None):
            cfg = compose(config_name=self._config_name, overrides=overrides)

        self._resolved_overrides = overrides

        return cfg

    async def setup(self, environment: BaseEnvironment) -> None:
        if not isinstance(environment, SSADockerEnvironment):
            raise TypeError(
                f"SSANative requires SSADockerEnvironment but got {type(environment).__name__}. "
                "Set environment type to 'ssa-docker' in your trial config."
            )


        # Determine the container workdir
        result = await environment.exec(command="pwd")
        workdir = (result.stdout or "").strip() or "/app"
        # Load SSA config
        self._cfg = self._load_ssa_config(workdir)

        environment._workdir = workdir
        environment._default_timeout = self._cfg.env.timeout
        environment._cfg = self._cfg

        # Run container setup commands from SSA config (e.g., install ripgrep)
        root_setup_cmds = []
        if hasattr(self._cfg.env, "docker") and hasattr(self._cfg.env.docker, "root_setup_commands"):
            root_setup_cmds = list(self._cfg.env.docker.root_setup_commands or [])

        for cmd in root_setup_cmds:
            LOG.info(f"Running container setup: {cmd}")
            result = await environment.exec(
                command=cmd,
                env={"DEBIAN_FRONTEND": "noninteractive"},
            )
            if result.return_code != 0:
                LOG.warning(
                    f"Container setup command failed (rc={result.return_code}): {cmd}\n"
                    f"stderr: {result.stderr}"
                )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._cfg is None:
            raise RuntimeError("setup() must be called before run()")
        if not isinstance(environment, SSADockerEnvironment):
            raise TypeError(
                f"SSANative requires SSADockerEnvironment but got {type(environment).__name__}."
            )

        # SSA's agent loop is sync and long-running. Run it in a worker thread
        # so the event loop stays responsive (timeouts, progress updates, other
        # trials). No async/sync bridge is needed for environment calls — they
        # use subprocess.run() directly.
        #
        # When harbor's wait_for times out, CancelledError is raised here at the
        # await. The finally block signals the thread to stop via _stop_event;
        # the callback handler checks this and raises AgentStoppedError to abort
        # the strands agent loop from within.
        self._stop_event.clear()
        try:
            metrics_data = await asyncio.to_thread(
                self._run_ssa, instruction, environment
            )
        except asyncio.CancelledError:
            self._stop_event.set()
            raise
        except BaseException:
            self._stop_event.set()
            raise

        context.n_input_tokens = metrics_data.get("input_tokens")
        context.n_output_tokens = metrics_data.get("output_tokens")
        context.n_cache_tokens = metrics_data.get("cache_tokens")
        cost = metrics_data.get("cost_usd", 0)
        context.cost_usd = cost if cost and cost > 0 else None
        context.metadata = metrics_data.get("metadata")

    def _run_ssa(self, instruction: str, environment: SSADockerEnvironment) -> dict:
        """Run the SSA agent loop. All sync — no threads needed."""
        from strands.handlers.callback_handler import (
            CompositeCallbackHandler,
        )

        import ssa.utils.monkey_patch  # noqa: F401
        from ssa.agent import StrandsResolverAgent
        from ssa.agent_runner import initialize_hooks, sr_model
        from ssa.callbacks.throttling import ThrottlingCallback
        from ssa.conversation_manager.conversation_manager import (
            AdaptiveConversationManager,
        )
        from ssa.metrics import MetricsCollector
        from ssa.prompts.prompt_gen import PromptGenerator
        from ssa.tools import load_tools

        cfg = self._cfg
        output_dir = str(self.logs_dir / "ssa-output")
        os.makedirs(output_dir, exist_ok=True)

        # Manually dump configs since compose() doesn't create .hydra/ like @hydra.main
        hydra_dir = os.path.join(output_dir, ".hydra")
        os.makedirs(hydra_dir, exist_ok=True)
        with open(os.path.join(hydra_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        with open(os.path.join(hydra_dir, "overrides.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self._resolved_overrides))

        # Set up file logging since we use hydra.compose() instead of @hydra.main
        # (compose doesn't configure logging/output dirs — only loads config).
        #
        # Tag this trial with a unique contextvar so the file handler's filter
        # only accepts log records from threads belonging to THIS trial.
        # strands' run_async() and asyncio.to_thread() copy contextvars,
        # so child threads automatically inherit the tag.
        trial_id = str(uuid.uuid4())
        _TRIAL_ID.set(trial_id)

        log_file_path = os.path.join(output_dir, "agent.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        file_handler.addFilter(_TrialFilter(trial_id))

        # Attach to loggers used by SSA and strands.  The filter ensures only
        # records from THIS trial's threads pass through to this file.
        _library_logger_names = [
            "ssa",
            "strands",
            "utils.harbor_plugin.ssa_docker",
            "utils.harbor_plugin.ssa_native",
        ]
        _library_loggers = []
        for name in _library_logger_names:
            lg = logging.getLogger(name)
            lg.addHandler(file_handler)
            if lg.level > logging.INFO or lg.level == logging.NOTSET:
                lg.setLevel(logging.INFO)
            _library_loggers.append(lg)

        # Suppress noisy OTel context-detach errors that fire when
        # AgentStoppedError unwinds through strands' generator stack.
        otel_logger = logging.getLogger("opentelemetry.context")
        otel_shutdown_filter = _OtelShutdownFilter(self._stop_event)
        otel_logger.addFilter(otel_shutdown_filter)

        # Generate prompts
        prompts_dir = str(Path(__import__("ssa").__file__).parent / "prompts")
        prompt_generator = PromptGenerator(base_dir=prompts_dir)

        project_path = environment.workdir
        agent_name = cfg.agent.agent_id
        prompt_tag = cfg.agent.prompt_tag

        system_prompt = prompt_generator.get_system_prompt(
            agent_name, prompt_tag=prompt_tag, project_path=project_path
        )
        user_prompt = prompt_generator.get_user_prompt(
            agent_name,
            prompt_tag=prompt_tag,
            project_path=project_path,
            git_issue=instruction,
        )

        LOG.info(f"System prompt length: {len(system_prompt)} chars")
        LOG.info(f"User prompt length: {len(user_prompt)} chars")

        LOG.info(f"system prompt:\n{system_prompt}")
        LOG.info(f"user prompt:\n{user_prompt}")

        # Initialize conversation manager, callbacks, hooks
        conversation_manager = AdaptiveConversationManager(
            window_size=150 * 2,
            should_truncate_results=True,
            per_turn=True,
        )
        callback_handler = CompositeCallbackHandler(
            ThrottlingCallback(),
            _LoggingCallbackHandler(stop_event=self._stop_event),
        )
        hooks = initialize_hooks(cfg, output_dir)

        # Create model (LLM calls happen on the host — no container resource limits)
        model = sr_model(cfg)

        # Load tools
        tools, tool_params = load_tools(cfg)

        with MetricsCollector(output_dir) as mc:
            sra = StrandsResolverAgent(
                system_prompt=system_prompt,
                model=model,
                tools=tools,
                conversation_manager=conversation_manager,
                hooks=hooks,
                callback_handler=callback_handler,
            )
            mc.bind(sra)
            sra_result = None
            try:
                while not self._stop_event.is_set():
                    sra_result = sra(
                        user_prompt,
                        environment=environment,
                        show_panel=False,
                        tool_params=tool_params,
                    )
                    retry_feedback = environment.retry_feedback(sra, sra_result)
                    if retry_feedback:
                        user_prompt = retry_feedback
                    else:
                        break

                LOG.info("SSA agent execution completed")
            finally:
                otel_logger.removeFilter(otel_shutdown_filter)
                for lg in _library_loggers:
                    lg.removeHandler(file_handler)
                file_handler.close()

                mc.dump(sra)

                # Extract metrics
                metrics = sra.event_loop_metrics
                accumulated = dict(metrics.accumulated_usage) if metrics.accumulated_usage else {}

                return {
                    "input_tokens": accumulated.get("inputTokens"),
                    "output_tokens": accumulated.get("outputTokens"),
                    "cache_tokens": accumulated.get("cacheReadInputTokens"),
                    "cost_usd": None,
                    "metadata": {
                        "config_name": self._config_name,
                        "model": str(cfg.agent.model),
                        "accumulated_usage": accumulated,
                    },
                }
