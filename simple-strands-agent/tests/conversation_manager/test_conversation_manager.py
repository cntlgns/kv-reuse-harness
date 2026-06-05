"""Tests for ssa.conversation_manager.AdaptiveConversationManager."""

from __future__ import annotations

from dataclasses import dataclass

from ssa.conversation_manager.conversation_manager import AdaptiveConversationManager


@dataclass
class _FakeAgent:
    """Minimal stand-in for strands.Agent — only ``messages`` is ever read."""

    messages: list


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


def _assistant_tool_use(tool_use_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": "bash", "input": {}}}],
    }


def _user_tool_result(tool_use_id: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "status": "success",
                    "content": [{"text": text}],
                }
            }
        ],
    }


def test_apply_management_noop_under_window():
    """Below window_size, apply_management doesn't mutate messages."""
    cm = AdaptiveConversationManager(window_size=10)
    messages = [_user("hi"), _assistant_text("hello"), _user("bye")]
    agent = _FakeAgent(messages=messages)
    before = list(messages)

    cm.apply_management(agent)

    assert agent.messages == before


def test_reduce_from_overflow_truncates_largest_tool_result():
    """With from_overflow=True, the message with the largest tool-result payload
    is truncated, and smaller tool-result messages are left alone."""
    cm = AdaptiveConversationManager(window_size=10)

    small_text = "short"
    big_text = "X" * 5000  # well over MAX_TOOL_OUTPUT_CHARS (100)

    messages = [
        _user("task"),
        _assistant_tool_use("tu1"),
        _user_tool_result("tu1", small_text),
        _assistant_tool_use("tu2"),
        _user_tool_result("tu2", big_text),
    ]
    agent = _FakeAgent(messages=messages)

    cm.reduce_context(agent, from_overflow=True)

    # Small tool result is unchanged
    small_block = agent.messages[2]["content"][0]["toolResult"]["content"][0]["text"]
    assert small_block == small_text

    # Big tool result got shrunk
    big_block = agent.messages[4]["content"][0]["toolResult"]["content"][0]["text"]
    assert len(big_block) < len(big_text)


def test_reduce_context_skips_orphan_toolresult_at_trim_index():
    """A naive trim_index landing on a ``toolResult`` message must advance past
    it so the remaining window never starts with an orphaned result."""
    cm = AdaptiveConversationManager(window_size=3)

    # 6 messages; with window_size=3 the default trim_index is len - window = 3.
    # Message at index 3 is a toolResult -> trim_index must advance.
    messages = [
        _user("task"),                              # 0 — kept (first)
        _assistant_tool_use("tu1"),                 # 1 — kept (first assistant tool-use, paired w/ result)
        _user_tool_result("tu1", "r1"),             # 2 — kept (its result)
        _user_tool_result("orphan", "should not start window"),  # 3 — skip (orphan result)
        _assistant_text("summary"),                 # 4 — valid trim target
        _user("next-question"),                     # 5
    ]
    agent = _FakeAgent(messages=messages)

    cm.reduce_context(agent)

    # Window must not start with the orphaned toolResult at old index 3.
    # The kept prefix is messages[:2+1] (first user+assistant-tool+result),
    # and the suffix starts at the first non-orphan index (4).
    assert all(
        not any("toolResult" in c for c in msg["content"] if isinstance(c, dict))
        or msg is messages[2]  # the only retained toolResult is the paired one at index 2
        for msg in agent.messages
    )
    # The window should contain the assistant summary and the follow-up user msg.
    roles = [m["role"] for m in agent.messages]
    texts = [
        c.get("text")
        for m in agent.messages
        for c in m["content"]
        if isinstance(c, dict) and "text" in c
    ]
    assert "summary" in texts
    assert "next-question" in texts
    assert roles[0] == "user"  # first message always preserved
