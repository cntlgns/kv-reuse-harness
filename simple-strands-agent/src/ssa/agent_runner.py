#!/usr/bin/env python3
"""
SSA Agent Runner - Environment creation and agent execution.

This module:
1. Initializes conversation manager and hooks
2. Creates and manages the execution environment (Docker or Local)
3. Instantiates the agent with provided configuration
4. Runs the agent within the environment context
5. Handles cleanup
"""

import logging
import os
from omegaconf import DictConfig

from strands.handlers.callback_handler import PrintingCallbackHandler, CompositeCallbackHandler
from ssa.agent import StrandsResolverAgent
from ssa.environments import create_environment
from ssa.conversation_manager.conversation_manager import AdaptiveConversationManager
from ssa.hooks import initialize_hooks
from ssa.callbacks.throttling import ThrottlingCallback
from ssa.models import sr_model
from ssa.tools import load_tools
from ssa.metrics import MetricsCollector
LOG = logging.getLogger(__name__)


def run_agent(
    cfg: DictConfig,
    system_prompt: str,
    user_prompt: str,
    output_dir: str
) -> None:
    """
    Create environment and run agent.

    Args:
        cfg: Hydra DictConfig
        system_prompt: Generated system prompt
        user_prompt: Generated user prompt
        output_dir: Hydra output directory
    """
    LOG.info("Starting agent runner")
    LOG.info(f"Environment type: {cfg.env.env_type}")
    LOG.info(f"system prompt:\n{system_prompt}")
    LOG.info(f"user_prompt:\n{user_prompt}")

    # Initialize conversation manager
    win_len = 150*2 # Default: 150 User-Assist pair unless overwrite by cfg
    if hasattr (cfg.agent, "conversation_manager"):
        win_len = cfg.agent.conversation_manager.get("win_len", 150*2)
    conversation_manager = AdaptiveConversationManager(
        window_size=win_len,
        should_truncate_results=True,
        per_turn=True,
    )
    callback_handler = CompositeCallbackHandler(
        PrintingCallbackHandler(),
        ThrottlingCallback(),
    )

    # Initialize hooks
    hooks = initialize_hooks(cfg, output_dir)

    with create_environment(cfg, output_dir) as env, MetricsCollector(output_dir) as mc:
        LOG.info("Environment initialized successfully")

        tools, tool_params = load_tools(cfg)
        sra = StrandsResolverAgent(
            system_prompt=system_prompt,
            model=sr_model(cfg),
            tools=tools,
            conversation_manager=conversation_manager,
            hooks=hooks,
            callback_handler=callback_handler,
        )
        mc.bind(sra)
        # Task identity for per-request telemetry (see SROpenAIModel request_log):
        # arm comes from the launcher via SSA_ARM since it is a hydra-override
        # combination with no config key of its own.
        request_meta = {
            "bench": str(cfg.dataset.get("name", "na")),
            "arm": os.environ.get("SSA_ARM", "na"),
            "task_id": str(cfg.dataset.get("identifier", "na")),
            "output_dir": output_dir,
        }
        sra_result = None
        try:
            # Run agent
            while True:
                sra_result = sra(user_prompt,
                    environment=env,
                    show_panel=True,
                    tool_params=tool_params,
                    request_meta=request_meta,
                )
                retry_feedback = env.retry_feedback(sra, sra_result)
                if retry_feedback:
                    user_prompt = retry_feedback
                else:
                    break
            LOG.info("Agent execution completed")
        except (KeyboardInterrupt, SystemExit):
            LOG.warning("Agent interrupted; dumping metrics before exit...")
        finally:
            env.collect_submission(agent=sra, result=sra_result)
            mc.dump(sra)

    LOG.info("Environment cleaned up successfully")
