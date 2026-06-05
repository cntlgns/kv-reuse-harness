"""Tests for ssa.tools.bash."""

from __future__ import annotations

import pytest

from ssa.tools.bash import bash
from tests.conftest import FakeEnvironment


def _invoke(tool_input: dict, env: FakeEnvironment, **kwargs) -> dict:
    return bash(
        {"toolUseId": "t1", "input": tool_input},
        environment=env,
        show_panel=False,
        **kwargs,
    )


def test_bash_missing_command(fake_env):
    """Missing ``command`` returns an error result."""
    result = _invoke({}, fake_env)

    assert result["status"] == "error"
    assert "Command is required" in result["content"][0]["text"]
    # No command should have been dispatched to the env
    assert fake_env.calls == []


@pytest.mark.parametrize("publish_partial", [True, False])
def test_bash_timeout_124_message_and_partial(fake_env, publish_partial):
    """Exit code 124 prepends the timeout notice and keeps partial output
    unless publish_partial_output is disabled."""
    fake_env.queue_response(
        exit_code=124, status="error", output="partial-stdout", error=""
    )

    result = bash(
        {"toolUseId": "t1", "input": {"command": "sleep 10", "timeout": 1}},
        environment=fake_env,
        show_panel=False,
        tool_params={"bash": {"publish_partial_output": publish_partial}},
    )

    text = result["content"][0]["text"]
    assert "Command timed-out with limit" in text
    assert "Exit Code: 124" in text
    if publish_partial:
        assert "partial-stdout" in text
    else:
        # Output line is still present but the captured payload is cleared.
        assert "partial-stdout" not in text


def test_bash_output_clipping(fake_env):
    """Outputs exceeding MAX_LINES_LIMIT (250) are head/tail clipped with a
    marker indicating how many lines were dropped."""
    big_output = "\n".join(f"line-{i}" for i in range(500))
    fake_env.queue_response(output=big_output, exit_code=0, status="success")

    result = _invoke({"command": "seq 500"}, fake_env)

    text = result["content"][0]["text"]
    assert "lines clipped" in text
    # Head is preserved
    assert "line-0" in text
    # Tail is preserved
    assert "line-499" in text
    # Something in the middle got dropped
    assert "line-250" not in text
