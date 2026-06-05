"""
Terminal Bench 2 (TB2) utilities for evaluation and testing.

This module provides utilities for running TB2 evaluations on agent outputs.
"""

import logging
import os
import subprocess

from omegaconf import DictConfig

from ssa.environments.docker_utils import parse_task_toml_config

LOG = logging.getLogger(__name__)


def run_tb2_eval(cfg: DictConfig, env, output_dir: str) -> None:
    """
    Run TB2 evaluation by executing tests in the docker container.

    Args:
        cfg: Hydra DictConfig
        env: Environment instance with docker_connector
        output_dir: Hydra output directory for saving results

    The function:
    1. Copies test files from TB2 repo into docker container
    2. Executes test.sh in the container
    3. Saves test output to docker_test_output.log
    4. Copies reward.txt (0/1 result) from container to output directory
    """
    LOG.info("Running TB2 evaluation...")

    # Get TB2 repo path and test directory
    tb2_repo_path = os.environ.get("TB2_REPO_PATH")
    if not tb2_repo_path:
        raise ValueError("TB2_REPO_PATH environment variable not set. Please set it to the path of the TB2 repository.")
    tb2_test_dir = f"{tb2_repo_path}/{cfg.dataset.identifier}/tests"

    if not os.path.exists(tb2_test_dir):
        LOG.error(f"TB2 test directory not found: {tb2_test_dir}")
        raise FileNotFoundError(f"TB2 tests not found at {tb2_test_dir}")

    # Get docker container ID
    container_id = env.docker_connector.container.id
    LOG.info(f"Using docker container: {container_id[:12]}")

    # Copy test directory into docker container
    LOG.info(f"Copying tests from {tb2_test_dir} to container:/tests")
    subprocess.run(
        f"docker cp {tb2_test_dir}/. {container_id}:/tests",
        shell=True,
        check=True
    )

    # Get verifier timeout from task.toml
    tb2_task_toml = os.path.join(tb2_repo_path, cfg.dataset.identifier, "task.toml")
    task_config = parse_task_toml_config(tb2_task_toml)
    verifier_timeout = task_config['verifier_timeout_sec']

    LOG.info(f"Running TB2 tests with timeout: {verifier_timeout}s")

    # Execute tests in container
    exit_code, output = env.docker_connector.exec(
        "bash /tests/test.sh",
        cwd="",
        timeout_sec=verifier_timeout,
        verbose=True
    )

    LOG.info(f"TB2 test completed with exit_code: {exit_code}")
    LOG.info(f"Test output:\n{output}")

    # Write test output to log file
    test_output_path = os.path.join(output_dir, "docker_test_output.log")
    with open(test_output_path, "w") as f:
        f.write(f"exit_code: {exit_code}\n{output}")
    LOG.info(f"Saved test output to {test_output_path}")

    # Copy reward.txt from container to output directory
    # TB2 tests write 0 (fail) or 1 (pass) to /logs/verifier/reward.txt
    docker_reward_path = "/logs/verifier/reward.txt"
    local_reward_path = os.path.join(output_dir, "reward.txt")

    try:
        subprocess.run(
            f"docker cp {container_id}:{docker_reward_path} {local_reward_path}",
            shell=True,
            check=True
        )
        LOG.info(f"Copied reward.txt to {local_reward_path}")

        # Read and log the reward
        with open(local_reward_path, "r") as f:
            reward = f.read().strip()
        LOG.info(f"TB2 Evaluation Result: {reward} ({'PASS' if reward == '1' else 'FAIL'})")

    except subprocess.CalledProcessError as e:
        LOG.warning(f"Failed to copy reward.txt: {e}")
        LOG.warning("Writing default reward (0) to file")
        with open(local_reward_path, "w") as f:
            f.write("0\n")

    LOG.info("TB2 evaluation completed")
