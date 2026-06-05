import logging
import re

from strands.hooks import (
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)


LOG = logging.getLogger(__name__)


# Patterns that indicate strands synthesized an error result for a bad tool call
# rather than executing one. Covers:
#   - Unknown tool (selected_tool was None in BeforeToolCallEvent)
#   - Invalid tool name pattern (rejected upstream by validate_and_prepare_tools)
_BAD_TOOL_PATTERNS = (
    re.compile(r"^Unknown tool: "),
    re.compile(r"\binvalid tool name pattern\b"),
    re.compile(r"\binvalid tool name length\b"),
    re.compile(r"\btool name missing\b"),
    re.compile(r"\btool cancelled by user\b"),
)


def _is_bad_tool_result(tr: dict) -> bool:
    if tr.get("status") != "error":
        return False
    for block in tr.get("content", []):
        text = block.get("text", "") if isinstance(block, dict) else ""
        if any(p.search(text) for p in _BAD_TOOL_PATTERNS):
            return True
    return False


class DropUnknownToolHook(HookProvider):
    """Rewind the conversation when the model emits a malformed tool call.

    Covers three paths:
      1. BeforeToolCallEvent with selected_tool=None — tool name passes
         regex but isn't registered. Cancels execution preemptively so we
         don't run anything for this turn.
      2. BeforeToolCallEvent with empty input ({} / no keys) — the model
         emitted a tool call shell with no arguments. Cancel preemptively
         so we don't invoke a real tool with garbage.
      3. MessageAddedEvent for the user toolResult message — catches the
         above plus the upstream validator path (invalid name pattern,
         where the tool_use never even reaches the executor). If ANY
         toolResult in the message is a synthesized bad-tool error, the
         whole turn is treated as corrupted: drop that user message AND
         the assistant message above it.
    """

    def __init__(self) -> None:
        self._pending_drop: bool = False

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)
        registry.add_callback(MessageAddedEvent, self._on_message_added)

    def _on_before_tool_call(self, event: BeforeToolCallEvent):
        if self._pending_drop:
            event.cancel_tool = True
            return

        tool_name = event.tool_use.get("name")

        if event.selected_tool is None:
            LOG.warning(
                "Unknown tool '%s' — will drop assistant message and retry model",
                tool_name,
            )
            event.cancel_tool = True
            self._pending_drop = True
            return

        tool_input = event.tool_use.get("input")
        if not tool_input:
            LOG.warning(
                "Tool '%s' called with empty input — will drop assistant message and retry model",
                tool_name,
            )
            event.cancel_tool = True
            self._pending_drop = True

    def _on_message_added(self, event: MessageAddedEvent):
        msg = event.message
        if msg.get("role") != "user":
            return

        tool_results = [
            c["toolResult"] for c in msg.get("content", []) if "toolResult" in c
        ]
        if not tool_results:
            return

        # If any toolResult in this turn is a synthesized bad-tool error,
        # treat the entire turn as corrupted and rewind.
        if not any(_is_bad_tool_result(tr) for tr in tool_results):
            self._pending_drop = False
            return

        messages = event.agent.messages
        # Pop the just-added user toolResult message
        if messages and messages[-1] is msg:
            messages.pop()
        # Pop the assistant message that produced the bad tool_use(s)
        if messages and messages[-1].get("role") == "assistant":
            messages.pop()
        LOG.warning("Dropped assistant+toolResult turn due to bad tool call(s)")
        self._pending_drop = False
