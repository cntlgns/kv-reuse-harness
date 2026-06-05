"""End-to-end tests wiring FakeSSAModel into a real SSA agent run.

These exercise:
  - agent → model → tool → env → trajectory hook loop
  - agent's throttling-retry path via ContentHook
"""

from __future__ import annotations

import json
from pathlib import Path


from ssa.agent import StrandsResolverAgent
from ssa.environments.environment import LocalEnvironment
from ssa.hooks.content_hook import ContentHook
from ssa.hooks.traj_hook import TrajectoryHook
from ssa.tools import bash as bash_module
from ssa.tools import submit as submit_module
from tests.conftest import make_cfg
from tests.fake_model import FakeSSAModel, FakeTurn


def _make_local_env(workdir: Path) -> LocalEnvironment:
    cfg = make_cfg(env={"env_type": "local", "timeout": 30, "local": {"workdir": str(workdir)}})
    return LocalEnvironment(cfg, output_dir=str(workdir))


def test_e2e_local_agent_runs_to_submit(tmp_path):
    """Agent runs a bash command, submits, trajectory.json reflects the conversation."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    target = workdir / "hello.txt"
    target.write_text("before\n")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    model = FakeSSAModel(
        [
            FakeTurn(
                text="I will update the file.",
                tool_calls=[
                    (
                        "bash",
                        "tu1",
                        {
                            "description": "write into the file",
                            "command": f"echo updated > {target}",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            FakeTurn(
                text="Done.",
                tool_calls=[
                    (
                        "submit",
                        "tu2",
                        {
                            "summary": "wrote hello.txt",
                            "status": "success",
                            "paths": [str(target)],
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
        ]
    )

    traj_hook = TrajectoryHook(output_dir=str(output_dir), record_interval=1)
    env = _make_local_env(workdir)

    with env:
        agent = StrandsResolverAgent(
            system_prompt="test",
            model=model,
            tools=[bash_module, submit_module],
            hooks=[traj_hook],
        )
        result = agent(
            "Please update the file and submit.",
            environment=env,
            show_panel=False,
            tool_params={},
        )

    # The submit tool requested a clean stop via request_state
    assert result.state.get("stop_event_loop") is True
    assert result.state.get("submit_paths") == [str(target)]

    # Trajectory was written
    traj_path = output_dir / "trajectory.json"
    assert traj_path.exists()
    traj = json.loads(traj_path.read_text())
    assert isinstance(traj, list) and len(traj) >= 4
    roles = [m["role"] for m in traj]
    # Expect: user → assistant (bash) → user (tool result) → assistant (submit) → user (tool result)
    assert roles[0] == "user"
    assert "assistant" in roles
    # And the bash side-effect actually happened
    assert target.read_text().strip() == "updated"


def test_e2e_throttling_retry_path(tmp_path):
    """ContentHook converts an empty-content assistant message into a throttle;
    the agent retries and the second turn produces a valid submit."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    target = workdir / "file.txt"
    target.write_text("")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Turn 1: empty content on end_turn → ContentHook throttles
    # Turn 2: clean submit
    model = FakeSSAModel(
        [
            FakeTurn(text=None, tool_calls=[], stop_reason="end_turn"),
            FakeTurn(
                text="Done after retry.",
                tool_calls=[
                    (
                        "submit",
                        "tu1",
                        {
                            "summary": "done",
                            "status": "success",
                            "paths": [str(target)],
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
        ]
    )

    content_hook = ContentHook()
    traj_hook = TrajectoryHook(output_dir=str(output_dir), record_interval=1)
    env = _make_local_env(workdir)

    with env:
        agent = StrandsResolverAgent(
            system_prompt="test",
            model=model,
            tools=[submit_module],
            hooks=[content_hook, traj_hook],
        )
        result = agent(
            "Please submit.",
            environment=env,
            show_panel=False,
            tool_params={},
        )

    # Model was called more than once — retry actually happened
    assert model.call_count >= 2
    # And final state shows the successful submit
    assert result.state.get("stop_event_loop") is True
    assert result.state.get("submit_paths") == [str(target)]
