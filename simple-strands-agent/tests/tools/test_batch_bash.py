"""Tests for ssa.tools.batch_bash."""

from __future__ import annotations

from ssa.tools.batch_bash import batch_bash


def test_batch_bash_ignore_errors_false_stops(fake_env):
    """With ignore_errors=False, a failure halts execution and remaining commands
    are annotated as skipped."""
    # cmd1 succeeds, cmd2 fails, cmd3 would succeed but should never run
    fake_env.queue_response(output="ok1", exit_code=0, status="success")
    fake_env.queue_response(output="boom", exit_code=1, status="error", error="boom")
    fake_env.queue_response(output="should-not-appear", exit_code=0, status="success")

    result = batch_bash(
        {
            "toolUseId": "t1",
            "input": {
                "commands": [
                    {"command": "cmd1", "description": "one", "ignore_errors": False},
                    {"command": "cmd2", "description": "two", "ignore_errors": False},
                    {"command": "cmd3", "description": "three", "ignore_errors": False},
                ]
            },
        },
        environment=fake_env,
        show_panel=False,
    )

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "ok1" in text
    assert "boom" in text
    assert "1 remaining command(s) skipped due to failure" in text
    assert "should-not-appear" not in text
    # Only two commands were dispatched to the environment
    assert len(fake_env.calls) == 2
