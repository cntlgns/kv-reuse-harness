"""Tests for ssa.hooks.content_hook.ContentHook."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from strands.hooks import AfterModelCallEvent
from strands.types.exceptions import ModelThrottledException

from ssa.hooks.content_hook import ContentHook, EventLoopLimiterHook
from ssa.types.exceptions import MaxRecursionsReachedException


def _event(message: dict, stop_reason: str, agent=None) -> AfterModelCallEvent:
    agent = agent or MagicMock()
    agent.messages = getattr(agent, "messages", [])
    return AfterModelCallEvent(
        agent=agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message=message, stop_reason=stop_reason
        ),
    )


@pytest.mark.parametrize("stop_reason", ["end_turn", "tool_use", "max_tokens"])
def test_content_hook_empty_content_throttles(stop_reason):
    """Empty content → ModelThrottledException regardless of stop_reason."""
    hook = ContentHook()
    event = _event({"role": "assistant", "content": []}, stop_reason)

    with pytest.raises(ModelThrottledException):
        hook.on_undesired_message_add(event)


def test_content_hook_max_tokens_with_tooluse_recovers():
    """``max_tokens`` with a toolUse in content → recovered message + tool_result
    feedback appended to agent.messages, throttling raised."""
    hook = ContentHook()
    agent = MagicMock()
    agent.messages = []

    message = {
        "role": "assistant",
        "content": [
            {"text": "trying to run something"},
            {
                "toolUse": {
                    "toolUseId": "tu1",
                    "name": "bash",
                    "input": {"command": "echo hi"},
                }
            },
        ],
    }
    event = _event(message, "max_tokens", agent=agent)

    with pytest.raises(ModelThrottledException, match="Max tokens reached"):
        hook.on_undesired_message_add(event)

    # Hook appended the recovered assistant message and a user feedback message.
    assert len(agent.messages) == 2
    appended_assistant = agent.messages[0]
    appended_user = agent.messages[1]
    assert appended_assistant["role"] == "assistant"
    # Recovered message should no longer contain any toolUse blocks
    assert not any(
        isinstance(c, dict) and "toolUse" in c for c in appended_assistant["content"]
    )
    assert appended_user["role"] == "user"
    assert "max_tokens" in appended_user["content"][0]["text"]


@pytest.mark.parametrize(
    "text",
    [
        "<ctrl46>",  # exact-pattern match
        "<ctrl46><ctrl46><ctrl46>spam",  # > 2 occurrences
        "call:default_api:something<ctrl46>",  # gemini tool-call-in-text regex
    ],
)
def test_content_hook_gemini_patterns_throttle(text):
    """Gemini-specific garbage patterns on end_turn → throttling."""
    hook = ContentHook()
    message = {"role": "assistant", "content": [{"text": text}]}
    event = _event(message, "end_turn")

    with pytest.raises(ModelThrottledException):
        hook.on_undesired_message_add(event)


def test_event_loop_limiter_enforces_limits():
    """Recursion and loop counters trip their respective limits."""
    hook = EventLoopLimiterHook(max_recursion_length=2, max_loop_length=2)

    # check_recursion_depth: the *first* three calls succeed (counter increments
    # to 3, then the next call sees counter > 2 and raises).
    event = MagicMock()
    hook.check_recursion_depth(event)
    hook.check_recursion_depth(event)
    hook.check_recursion_depth(event)
    with pytest.raises(MaxRecursionsReachedException):
        hook.check_recursion_depth(event)

    # check_event_loop raises once over limit (event.terminate is not yet
    # supported on AfterModelCallEvent, so the hook raises to break the loop).
    # Implementation increments *after* checking, so we need counter>max_loop.
    hook2 = EventLoopLimiterHook(max_recursion_length=100, max_loop_length=2)
    ev = MagicMock()
    for _ in range(3):
        hook2.check_event_loop(ev)  # counter climbs to 3, still <= limit
    with pytest.raises(Exception):
        hook2.check_event_loop(ev)  # now counter is 3 > 2 when checked
