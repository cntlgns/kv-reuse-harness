import logging
from typing import Optional, Any

from strands.agent import Agent
from strands.types.content import Messages
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    SlidingWindowConversationManager,
)
from strands.types.exceptions import ContextWindowOverflowException

LOG = logging.getLogger(__file__)

MAX_TOOL_OUTPUT_LINES: int = 2
MAX_TOOL_OUTPUT_CHARS: int = 100


class AdaptiveConversationManager(SlidingWindowConversationManager):
    def __init__(
        self,
        window_size: int = 40,
        should_truncate_results: bool = True,
        **kwargs,
        ):
        """Initialize the adaptive sliding window conversation manager.

        Args:
            window_size: Maximum number of messages to keep in history.
                Defaults to 40 messages.
        """
        super().__init__(window_size, should_truncate_results, **kwargs)

    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        messages = agent.messages

        if len(messages) > self.window_size:
            LOG.info(f"Message count={len(messages)} exceeding the window-len threshold={self.window_size}. Shadowing the large tool-results")
            self.reduce_context(agent)

    def reduce_context(self, agent: "Agent", e: Optional[Exception] = None, from_overflow: bool = False) -> None:
        """Trim the oldest messages to reduce the conversation context size.

        Keep the first 2 messages to maintain context of the conversation
        The method handles special cases where trimming the messages leads to:
         - toolResult with no corresponding toolUse
         - toolUse with no corresponding toolResult
        
         If context reduction is requested due to overflow, then shadow largest user-response

        Args:
            messages: The messages to reduce.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.

        Raises:
            ContextWindowOverflowException: If the context cannot be reduced further.
                Such as when the conversation is already minimal or when tool result messages cannot be properly
                converted.
        """
        # TODO: Tokens based context trimming, instead of a fixed window size
        # If the number of messages is less than the window_size, then we default to 2, otherwise, trim to window size
        messages = agent.messages

        if from_overflow:
        # Try to truncate the tool result first
            message_idx_with_tool_results = self._find_largest_message_with_large_tool_results(messages)

            if message_idx_with_tool_results is not None:
                LOG.debug(
                    "message_index=<%s> | found message with tool results at index",
                    message_idx_with_tool_results,
                )
                results_truncated = self._truncate_tool_results(
                    messages, message_idx_with_tool_results
                )
                if results_truncated:
                    LOG.debug(
                        "message_index=<%s> | tool results truncated",
                        message_idx_with_tool_results,
                    )
                    return

        trim_index = 2 if len(messages) <= self.window_size else len(messages) - self.window_size

        # Find the next valid trim_index
        while trim_index < len(messages):
            if (
                # Oldest message cannot be a toolResult because it needs a toolUse preceding it
                any("toolResult" in content for content in messages[trim_index]["content"])
                or (
                    # Oldest message can be a toolUse only if a toolResult immediately follows it.
                    any("toolUse" in content for content in messages[trim_index]["content"])
                    and trim_index + 1 < len(messages)
                    and not any(
                        "toolResult" in content for content in messages[trim_index + 1]["content"]
                    )
                )
            ):
                trim_index += 1
            else:
                break
        else:
            # If we didn't find a valid trim_index, then we throw
            raise ContextWindowOverflowException("Unable to trim conversation context!") from e

        # Overwrite message history
        # Keep first user-assistant conversation, unless assistant message is tool-call. In latter, append
        # tool-result as well
        begin_end_idx = 1
        while begin_end_idx < trim_index:
            if (
                # First assistant message, if toolUse, needs successive toolResult as well
                any("toolUse" in content for content in messages[begin_end_idx]["content"])
                and (
                    begin_end_idx + 1 < trim_index
                    and any(
                        "toolResult" in content
                        for content in messages[begin_end_idx + 1]["content"]
                    )
                )
            ):
                begin_end_idx += 1
            else:
                break
        LOG.info(f"Trimming message window with len={len(messages)} to [0,{begin_end_idx}] U [{trim_index},{len(messages)-1}]")
        if begin_end_idx < trim_index:
            messages[:] = messages[:begin_end_idx+1] + messages[trim_index:]
        else:
            # No trimming
            messages[:] = messages[:]
    
    def _find_first_message_with_large_tool_results(self, messages: Messages) -> Optional[int]:
        """Find the index of the last message containing tool results.

        This is useful for identifying messages that might need to be truncated to reduce context size.

        Args:
            messages: The conversation message history.

        Returns:
            Index of the last message with tool results, or None if no such message exists.
        """
        # Iterate forward through all messages (from oldest to newest)
        for idx in range(len(messages)):
            # Check if this message has any content with toolResult
            current_message = messages[idx]
            has_lengthy_tool_result = False

            for content in current_message.get("content", []):
                if isinstance(content, dict) and "toolResult" in content:
                    tool_content = content.get("toolResult", {}).get("content", [])
                    for _c in tool_content:
                        if isinstance(_c, dict) and "text" in _c:
                            if len(_c["text"]) > MAX_TOOL_OUTPUT_CHARS:
                                has_lengthy_tool_result = True
                                break

            if has_lengthy_tool_result:
                return idx

        return None
    
    def _find_largest_message_with_large_tool_results(self, messages: Messages) -> Optional[int]:
        """Find the index of the message containing tool results with largest char counts.

        This is useful for identifying messages that might need to be truncated to reduce context size.

        Args:
            messages: The conversation message history.

        Returns:
            Index of the last message with tool results, or None if no such message exists.
        """
        # Iterate forward through all messages (from oldest to newest)
        char_count = {}
        for idx in range(len(messages)):
            # Check if this message has any content with toolResult
            current_message = messages[idx]

            for content in current_message.get("content", []):
                if isinstance(content, dict) and "toolResult" in content:
                    tool_content = content.get("toolResult", {}).get("content", [])
                    for _c in tool_content:
                        if isinstance(_c, dict) and "text" in _c:
                            if idx in char_count:
                                char_count[idx] += len(_c["text"])
                            else:
                                char_count[idx] = len(_c["text"])


        _max_val = 0
        _max_val_idx = -1
        for idx, c_count in char_count.items():
            if c_count > _max_val:
                _max_val = c_count
                _max_val_idx = idx
        if _max_val_idx > -1:
            if char_count[_max_val_idx] > MAX_TOOL_OUTPUT_CHARS:
                return _max_val_idx

        return None
