import logging
from typing import List
from omegaconf import DictConfig
from ssa.hooks.content_hook import ContentHook, EventLoopLimiterHook
from ssa.hooks.traj_hook import TrajectoryHook
from ssa.hooks.caching_hook import PromptCacheHook
from ssa.hooks.unknown_tool_hook import DropUnknownToolHook
from ssa.hooks.harmony_retry_hook import HarmonyRetryHook
from ssa.hooks.context_window_hook import ContextWindowHook

LOG = logging.getLogger(__name__)


def initialize_hooks(cfg: DictConfig, output_dir: str) -> List:
    """
    Initialize hooks from config.

    Args:
        cfg: Hydra DictConfig

    Returns:
        List of hook instances
    """
    hooks = []

    content_hook = ContentHook()
    hooks.append(content_hook)
    LOG.info("Initialized ContentHook")

    event_loop_limiter = EventLoopLimiterHook(
        max_recursion_length=250,
        max_loop_length=cfg.max_llm_iterations,
    )
    hooks.append(event_loop_limiter)
    LOG.info("Initialized EventLoopLimiterHook")

    traj_hook = TrajectoryHook(output_dir=output_dir)
    hooks.append(traj_hook)
    LOG.info("Initialized Trajectory record hook")

    hooks.append(DropUnknownToolHook())
    LOG.info("Initialized DropUnknownToolHook")

    hooks.append(HarmonyRetryHook())
    LOG.info("Initialized HarmonyRetryHook")

    caching_params = cfg.agent.get("prompt_caching_params", {})
    _strategy = caching_params.get("strategy", "auto")
    if (
        cfg.agent.get("prompt_caching", False) and
        _strategy == "custom" and
        "anthropic" in cfg.agent.model.lower()
    ):
        p_cache_hook = PromptCacheHook(**caching_params)
        hooks.append(p_cache_hook)
        LOG.info("Initialized Prompt caching hook")

    ctx_window_params = cfg.agent.get("context_window", {}) or {}
    max_model_len = ctx_window_params.get("max_model_len")
    near_limit_threshold = ctx_window_params.get("near_limit_threshold")
    if max_model_len is not None and near_limit_threshold is not None:
        hooks.append(ContextWindowHook(
            max_model_len=max_model_len,
            near_limit_threshold=near_limit_threshold,
        ))
        LOG.info(
            f"Initialized ContextWindowHook (max_model_len={max_model_len}, "
            f"near_limit_threshold={near_limit_threshold})"
        )
    return hooks
