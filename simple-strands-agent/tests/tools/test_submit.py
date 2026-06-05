"""Tests for ssa.tools.submit."""

from __future__ import annotations

import pytest

from ssa.tools.submit import submit
from tests.conftest import FakeEnvironment


def _invoke(tool_input: dict, env: FakeEnvironment, request_state: dict | None = None) -> dict:
    return submit(
        {"toolUseId": "t1", "input": tool_input},
        environment=env,
        show_panel=False,
        request_state=request_state if request_state is not None else {},
    )


@pytest.mark.parametrize(
    "tool_input,expected_substr",
    [
        (
            {"status": "success", "paths": []},
            "summary parameter is required",
        ),
        (
            {"summary": "done", "paths": []},
            "status parameter is required",
        ),
        (
            {"summary": "done", "status": "success", "paths": "not-a-list"},
            "paths parameter is required and must be a list",
        ),
        (
            {"summary": "done", "status": "success", "paths": ["/missing/file"]},
            "The following paths do not exist",
        ),
    ],
)
def test_submit_validation(fake_env, tool_input, expected_substr):
    """Submit rejects missing/invalid fields and nonexistent paths."""
    result = _invoke(tool_input, fake_env)

    assert result["status"] == "error"
    assert expected_substr in result["content"][0]["text"]


def test_submit_success_sets_state():
    """A valid submit writes stop_event_loop and submit_paths into request_state."""
    env = FakeEnvironment(files={"/work/a.py": "", "/work/b.py": ""})
    state: dict = {}
    result = submit(
        {
            "toolUseId": "t1",
            "input": {
                "summary": "done",
                "status": "success",
                "paths": ["/work/a.py", "/work/b.py"],
            },
        },
        environment=env,
        show_panel=False,
        request_state=state,
    )

    assert result["status"] == "success"
    assert state["stop_event_loop"] is True
    assert state["submit_paths"] == ["/work/a.py", "/work/b.py"]
