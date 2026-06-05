#!/usr/bin/env python3
"""
SSA Main Entry Point - Configuration and Initialization.

This module:
1. Loads configuration from YAML using Hydra
2. Parses and prepares environment configuration
3. Initializes prompt generator and generates prompts
4. Delegates to agent_runner for execution
"""

import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ssa.environments.docker_utils import parse_docker_config
from ssa.prompts.prompt_gen import PromptGenerator
from ssa.agent_runner import run_agent
from ssa.utils.handle_config import load_problem_statement_from_config
from ssa.utils.signals import install_signal_handlers
import ssa.utils.monkey_patch  # noqa: F401 — applies litellm patches

LOG = logging.getLogger(__name__)


def generate_prompts(cfg: DictConfig, prompt_generator: PromptGenerator) -> tuple[str, str]:
    """
    Generate system and user prompts from config.

    Args:
        cfg: Hydra DictConfig
        prompt_generator: PromptGenerator instance

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    agent_name = cfg.agent.agent_id
    prompt_tag = cfg.agent.prompt_tag

    # Get project path (workdir)
    project_path = cfg.env.docker.workdir if cfg.env.env_type == "docker" else cfg.env.local.workdir

    # Load problem statement from dataset (tb2, sbv, or sbpro)
    try:
        git_issue = load_problem_statement_from_config(cfg)
        if not git_issue:
            # Fallback if already present in config
            git_issue = cfg.dataset.get("issue_description", "No issue description provided")
    except Exception as e:
        LOG.warning(f"Failed to load problem statement from dataset: {e}")
        raise ValueError(f"Failed to load problem statement from dataset: {e}") from e

    system_prompt = prompt_generator.get_system_prompt(
        agent_name,
        prompt_tag=prompt_tag,
        project_path=project_path
    )

    user_prompt = prompt_generator.get_user_prompt(
        agent_name,
        prompt_tag=prompt_tag,
        project_path=project_path,
        git_issue=git_issue
    )

    LOG.info(f"Generated prompts for agent: {agent_name}")
    LOG.info(f"System prompt length: {len(system_prompt)} chars")
    LOG.info(f"User prompt length: {len(user_prompt)} chars")

    return system_prompt, user_prompt


@hydra.main(version_base=None, config_path="configs", config_name="sbv_bedrock_opus_45_effort_high")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for SSA runner.

    Args:
        cfg: Hydra configuration
    """
    install_signal_handlers()
    LOG.info("Starting SSA Runner")
    LOG.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Get Hydra output directory
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    LOG.info(f"Hydra output directory: {output_dir}")

    # Parse and update environment config
    env_type = cfg.env.env_type
    LOG.info(f"Environment type: {env_type}")

    docker_config = None
    if env_type == "docker":
        LOG.info("Configuring Docker environment")
        docker_config = parse_docker_config(cfg)
        # Update cfg.env.docker with parsed values (base_image/workdir may change per dataset)
        cfg.env.docker.base_image = docker_config.base_image
        cfg.env.docker.workdir = docker_config.workdir
        OmegaConf.update(cfg, "env.host_mount", cfg.env.local.workdir, force_add=True)
    elif env_type == "local":
        LOG.info("Configuring Local environment")
    else:
        raise ValueError(f"Invalid env_type: {env_type}. Must be 'docker' or 'local'.")

    # Initialize prompt generator
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    prompt_generator = PromptGenerator(base_dir=prompts_dir)
    LOG.info(f"Initialized PromptGenerator with base_dir={prompts_dir}")

    # Generate prompts
    system_prompt, user_prompt = generate_prompts(cfg, prompt_generator)

    # Run agent with prepared configuration
    LOG.info("Delegating to agent_runner for execution")
    run_agent(
        cfg=cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_dir=output_dir
    )

    LOG.info("SSA Runner completed")


if __name__ == "__main__":
    main()
