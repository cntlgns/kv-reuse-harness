"""
Configuration handling utilities for SSA.

This module provides utilities for:
1. Loading dataset-specific problem statements
2. Mapping identifiers to problem descriptions
3. Dataset-specific configuration handling
"""

import json
import logging
import os
from datasets import load_dataset, load_from_disk
from typing import Optional

from omegaconf import DictConfig

LOG = logging.getLogger(__name__)


HF_SWE_VERIFIED: str = "princeton-nlp/SWE-bench_Verified"
HF_SWE_PRO: str = "ScaleAI/SWE-bench_Pro"


def get_problem_statement_for_tb2(identifier: str) -> str:
    """
    Get problem statement for TB2 dataset from instructions map.

    Args:
        identifier: TB2 project identifier (e.g., "adaptive-rejection-sampler")

    Returns:
        Problem statement string

    Raises:
        FileNotFoundError: If TB2_INSTRUCTIONS_MAP file not found
        KeyError: If identifier not found in instructions map
    """
    instructions_map_path = os.environ.get("TB2_INSTRUCTIONS_MAP")

    if not instructions_map_path:
        raise ValueError(
            "TB2_INSTRUCTIONS_MAP environment variable not set. "
            "Please set it to the path of the TB2 instructions JSON file."
        )

    if not os.path.exists(instructions_map_path):
        raise FileNotFoundError(
            f"TB2 instructions map not found at: {instructions_map_path}"
        )

    with open(instructions_map_path, "r") as f:
        instructions_map = json.load(f)

    if identifier not in instructions_map:
        raise KeyError(
            f"Could not find problem statement for TB2 project: {identifier}"
        )

    problem_statement = instructions_map[identifier]
    LOG.info(f"Loaded TB2 problem statement for {identifier} ({len(problem_statement)} chars)")

    return problem_statement

def get_problem_statement_for_swebench(dataset_name: str, identifier: str, use_hints: bool = False) -> str:
    """
    Get problem statement for SWEBench dataset.

    Args:
        identifier: SWEBench identifier (e.g., "django__django-15987")
        use_hints: Whether to include additional hints if available

    Returns:
        Problem statement string (with optional hints)

    Raises:
        FileNotFoundError: If SWEBENCH_DATA file not found
        KeyError: If identifier not found in data
    """
    if dataset_name == "sbv":
        cached_path = os.getenv("HF_SBV_DATASET_OFFLINE_LOCATION", "")
        if os.path.exists(cached_path):
            LOG.info(f"Using local cached hf dataset from path: {cached_path}")
            data= load_from_disk(cached_path)["test"]
        else:
            data = load_dataset(HF_SWE_VERIFIED, split="test")
        instance = data.filter(lambda x: x["instance_id"] == identifier)
        if len(instance) == 0:
            raise ValueError(f"Could not find swe-verified instance: {identifier}")
        problem_statement = instance[0]["problem_statement"]
        if use_hints:
            LOG.info(f"Loaded SWEBench problem statement with hints for {identifier}")
            problem_hints = instance[0].get("hints_text", "")
            problem_statement += (
                f"\n\nAdditional hints:\n{problem_hints}"
            )
    elif dataset_name == "sbpro":
        cached_path = os.getenv("HF_SBPRO_DATASET_OFFLINE_LOCATION", "")
        if os.path.exists(cached_path):
            LOG.info(f"Using local cached hf dataset from path: {cached_path}")
            data= load_from_disk(cached_path)["test"]
        else:
            data = load_dataset(HF_SWE_PRO, split="test")
        instance = data.filter(lambda x: x["instance_id"] == identifier)
        if len(instance) == 0:
            raise ValueError(f"Could not find swe-pro instance: {identifier}")
        problem_statement = instance[0]["problem_statement"].strip("\"")
        requirements = instance[0].get("requirements", "").strip("\"")
        new_interfaces = instance[0].get("interface", "").strip("\"")
        problem_statement += (
            (f"\n\nRequirements:\n{requirements}" if requirements else "")
            + (f"\n\nNew interfaces introduced:\n{new_interfaces}" if new_interfaces else "")
        )
    else:
        raise ValueError(f"Unsupported swe dataset: {dataset_name}")

    return problem_statement

def identifier_to_problem_statement(
    identifier: str,
    dataset_name: str,
    use_hints: bool = False
) -> str:
    """
    Map a dataset identifier to its problem statement.

    Args:
        identifier: Dataset-specific identifier
        dataset_name: Dataset type ("tb2", "sbv", or "sbpro")
        use_hints: For SWEBench, whether to include hints (default: False)

    Returns:
        Problem statement string

    Raises:
        ValueError: If dataset_name is not supported
        FileNotFoundError: If dataset file not found
        KeyError: If identifier not found in dataset
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "tb2":
        return get_problem_statement_for_tb2(identifier)
    elif dataset_name in ("sbv", "sbpro"):
        # Both SWEBench variants use the similar data source
        return get_problem_statement_for_swebench(dataset_name, identifier, use_hints=use_hints)
    else:
        raise ValueError(
            f"Unsupported dataset name: {dataset_name}. "
            "Supported values: 'tb2', 'sbv', 'sbpro'"
        )

def load_problem_statement_from_config(cfg: DictConfig) -> Optional[str]:
    """
    Load problem statement from config using identifier and dataset name.

    This is a convenience wrapper around identifier_to_problem_statement
    that extracts values from a DictConfig object.

    Args:
        cfg: Hydra DictConfig with dataset.identifier and dataset.name fields

    Returns:
        Problem statement string, or None if already present in cfg.dataset.issue_description

    Raises:
        ValueError: If required config fields missing
    """
    # If issue_description already set, return it
    if hasattr(cfg.dataset, "issue_description") and cfg.dataset.issue_description:
        LOG.info("Using existing issue_description from config")
        return cfg.dataset.issue_description

    # Extract required fields
    if not hasattr(cfg.dataset, "identifier"):
        raise ValueError("Missing cfg.dataset.identifier")

    if not hasattr(cfg.dataset, "name"):
        raise ValueError("Missing cfg.dataset.name")

    identifier = cfg.dataset.identifier
    dataset_name = cfg.dataset.name

    # Check for hints flag (SWEBench only)
    use_hints = os.environ.get("USE_SWEBENCH_HINTS", "false").lower() in ("true", "1", "yes")

    try:
        problem_statement = identifier_to_problem_statement(
            identifier=identifier,
            dataset_name=dataset_name,
            use_hints=use_hints
        )
        return problem_statement
    except Exception as e:
        LOG.error(f"Failed to load problem statement: {e}")
        raise
