import re
import uuid
import logging
from typing import List
from strands.event_loop._recover_message_on_max_tokens_reached import recover_message_on_max_tokens_reached
from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent
from strands.hooks import HookProvider, HookRegistry
from strands.types.exceptions import ModelThrottledException
from strands.types.event_loop import StopReason
from strands.types.content import Message
from strands.types.tools import ToolUse

from ssa.types.exceptions import MaxRecursionsReachedException


LOG = logging.getLogger(__name__)


EMPTY_STRINGS: List[str] = [
    "[blank text]",
]
GEMINI_TOOL_ENCODED_PATTERNS: List[str] = [
    r".*call:default_api:.*<ctrl46>",
]
GEMINI_UNWANTED_PATTERNS: List[str] = ["<ctrl46>",]
APPLY_PATCH_BEGIN_MARKER: str = "*** Begin Patch"
APPLY_PATCH_END_MARKER: str = "*** End Patch"


class ContentHook(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(AfterModelCallEvent, self.on_undesired_message_add)

    def on_undesired_message_add(self, event: AfterModelCallEvent):
        """
        Check for the following corner-cases:
        1. Invoker returned empty content list
            Treat this as an error
        2. Model responsed with CSM change blocks but without end_turn stop-reason
            Accept this message as end_turn and close conversation
        """
        agent = event.agent
        stop_response = event.stop_response
        if (
            stop_response is not None
            and isinstance(
                stop_response, 
                AfterModelCallEvent.ModelStopResponse
            )
        ):
            message = stop_response.message
            stop_reason: StopReason = stop_response.stop_reason
            content = message.get("content", [])
            if len(content) == 0:
                # Gateway invalid_prompt. A simple retry sends
                # the same body and may fail again — truncate the last turn
                # before raising
                model = getattr(agent, "model", None)
                if getattr(model, "_last_error_code", None) == "invalid_prompt":
                    err_msg = getattr(model, "_last_error_message", "") or ""
                    dropped = self._truncate_last_turn_for_invalid_prompt(agent)
                    model._last_error_code = None
                    model._last_error_message = None
                    e_message = (
                        f"Gateway returned invalid_prompt ({err_msg[:160]}). "
                        f"Dropped {dropped} trailing message(s) and retrying."
                    )
                    LOG.warning(e_message)
                    raise ModelThrottledException(message=e_message)
                e_message = f"Received empty content block from service with stop-reason={stop_reason}. Treating this as throttling"
                LOG.warning(e_message)
                raise ModelThrottledException(message=e_message)
            elif message.get("role") == "assistant":
                if stop_reason == "max_tokens":
                    if any(
                        "toolUse" in ct for ct in message["content"] 
                        if isinstance(ct, dict)
                    ):
                        LOG.info("Content-hook catching max-tokens exception")
                        message = recover_message_on_max_tokens_reached(message)
                        agent.messages.append(message)
                        # Original toolUse is removed from the message, provide a generic feedback about error input
                        # Create a feedback message
                        tool_fail_result = "The tool could not be executed due to max_tokens length"
                        tool_result_message: Message = {
                            "role": "user",
                            "content": [{"text": tool_fail_result}],
                        }
                        agent.messages.append(tool_result_message)
                        raise ModelThrottledException(message="Max tokens reached for tool-use. Re-trying after modifying the tool-use tool-result pair")

                if stop_reason != "end_turn":
                    for _, ct in enumerate(content):
                        if isinstance(ct, dict) and "text" in ct:
                            if self.has_blank_text(ct["text"]):
                                # check for garbage output from the model 
                                e_message = f"Received empty content block from service with stop-reason={stop_reason}. Treating this as throttling" 
                                LOG.warning(e_message)
                                raise ModelThrottledException(message=e_message)
                if stop_reason not in ("tool_use",):
                    for _, ct in enumerate(content):
                        if isinstance(ct, dict) and "toolUse" in ct:
                            # Check for unwanted tool-use assistant response without proper stop_reason
                            e_message = f"Received toolUse in the content with stop_reason={stop_reason}. Treating this as unexpected outcome and retrying with delay..."
                            LOG.warning(e_message)
                            raise ModelThrottledException(message=e_message)
                if (
                    stop_reason == "end_turn" and 
                    all(
                        self.has_blank_text(ct["text"])
                        for ct in content if isinstance(ct, dict) and "text" in ct
                    )
                ):
                    # All received text blocks are empty for the end_turn
                    e_message = "Received all empty text block from service. Treating this as throttling" 
                    LOG.warning(e_message)
                    raise ModelThrottledException(message=e_message)

                if (
                    stop_reason == "end_turn" and
                    any(
                        self.has_gemini_tool_call_in_text(ct["text"])
                        for ct in content if isinstance(ct, dict) and "text" in ct
                    )
                ):
                    # Received text blocks which are incorrect json parsing of tool-calls by Gemini API
                    e_message = f"Received tool-call json string in text block from service.\n {content}\nTreating this as throttling"
                    LOG.warning(e_message)
                    raise ModelThrottledException(message=e_message)
                if (
                    stop_reason == "end_turn" and
                    any(
                        self.has_gemini_unwanted_strings(ct["text"])
                        for ct in content if isinstance(ct, dict) and "text" in ct
                    )
                ):
                    # Received Gemini unwanted tokens in output text
                    e_message = f"Received Gemini unwanted tokens <ctrl46> in output `{content}`. Treating this as throttling"
                    LOG.warning(e_message)
                    raise ModelThrottledException(message=e_message)

                if (
                    stop_reason == "end_turn" and
                    any(
                        self.has_apply_patch_in_text(ct["text"])
                        for ct in content if isinstance(ct, dict) and "text" in ct
                    )
                ):
                    # Model emitted a patch as plain text instead of via the apply_patch tool.
                    # Invoke apply_patch inline and retry with the result fed back to the model.
                    self._recover_apply_patch_in_text(event, agent, message, content)

                if stop_reason == "content_filtered":
                    e_message = "Received content_filter trigger on model-output. Treating this as unexpected outcome and retrying with delay..."
                    LOG.warning(e_message)
                    raise ModelThrottledException(message=e_message)

    def _truncate_last_turn_for_invalid_prompt(self, agent) -> int:
        """Drop the trailing (assistant, user-tool_result) pair that triggered
        a gateway invalid_prompt parse error so the retry sends a shorter
        conversation.

        Returns the number of messages dropped (0–2).
        """
        msgs = agent.messages
        dropped = 0
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
            dropped += 1
        if msgs and msgs[-1].get("role") == "assistant":
            msgs.pop()
            dropped += 1
        return dropped

    def has_blank_text(self, content: str) -> bool:
        if len(content) == 0:
            return True
        blank_test = False
        for empty_str in EMPTY_STRINGS:
            if content == empty_str:
                blank_test = True
                break
        return blank_test
    
    def has_gemini_unwanted_strings(self, content: str) -> bool:
        for pattern in GEMINI_UNWANTED_PATTERNS:
            _content = content.strip(pattern)
            if len(_content) == 0: # output is exactly the unwanted pattern
                return True
            if content.count(pattern) > 2: # output has significant number of unwanted pattern along with other text
                return True
        return False            

    def has_gemini_tool_call_in_text(self, content: str) -> bool:
        for tool_pattern in GEMINI_TOOL_ENCODED_PATTERNS:
            if re.match(tool_pattern, content):
                return True
        return False

    def has_apply_patch_in_text(self, content: str) -> bool:
        return APPLY_PATCH_BEGIN_MARKER in content and APPLY_PATCH_END_MARKER in content

    def _recover_apply_patch_in_text(self, event, agent, message, content) -> None:
        # Lazy import to avoid pulling apply_patch (and Environment) at module import.
        from ssa.tools.openai.apply_patch import apply_patch as _apply_patch_handler

        patch_body = None
        for ct in content:
            if not (isinstance(ct, dict) and "text" in ct):
                continue
            text = ct["text"]
            start = text.find(APPLY_PATCH_BEGIN_MARKER)
            if start == -1:
                continue
            end = text.find(APPLY_PATCH_END_MARKER, start)
            if end == -1:
                continue
            patch_body = text[start:end + len(APPLY_PATCH_END_MARKER)]
            break
        if patch_body is None:
            return

        invocation_state = event.invocation_state or {}
        tool_params = dict(invocation_state.get("tool_params") or {})
        if not tool_params.get("openai.apply_patch"):
            for fallback_key in ("openai.v1.bash", "openai.v1.shell"):
                if tool_params.get(fallback_key):
                    tool_params["openai.apply_patch"] = tool_params[fallback_key]
                    break
        handler_kwargs = {**invocation_state, "tool_params": tool_params}

        tool_use_id = f"apply_patch_recovery_{uuid.uuid4().hex[:8]}"
        patch_tool: ToolUse = {
            "toolUseId": tool_use_id,
            "name": "apply_patch",
            "input": {
                "patch": patch_body,
                "description": "auto-recovered apply_patch from plaintext model output",
            },
        }

        patch_result = _apply_patch_handler(patch_tool, **handler_kwargs)
        result_text = "".join(
            b.get("text", "")
            for b in (patch_result.get("content") or [])
            if isinstance(b, dict)
        )
        status = patch_result.get("status", "unknown")
        LOG.info(f"Auto-invoked apply_patch from plaintext output (status={status})")

        agent.messages.append(message)
        feedback = (
            f"Detected apply_patch emitted as plain text "
            f"instead of the <apply_patch_command> via shell tool. Auto-invoking apply_patch on the block "
            f"(status={status}). Result:\n{result_text}\n"
            f"Please call the <apply_patch_command> via shell tool directly going forward."
        )
        agent.messages.append({"role": "user", "content": [{"text": feedback}]})

        e_message = (
            f"Recovered apply_patch from plaintext (status={status}). "
            "Triggering retry with augmented history."
        )
        LOG.warning(e_message)
        raise ModelThrottledException(message=e_message)


class EventLoopLimiterHook(HookProvider):
    def __init__(
        self,
        max_recursion_length: int = 250,
        max_loop_length: int = 500,
    ):
        super().__init__()
        self.max_recursion_length = max_recursion_length
        self.max_loop_length = max_loop_length
        self.loop_counter = 0
        self.recursion_counter = 0

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(AfterModelCallEvent, self.check_event_loop)
        registry.add_callback(BeforeModelCallEvent, self.check_recursion_depth)
    
    def check_recursion_depth(self, event: BeforeModelCallEvent) -> None:
        if self.recursion_counter > self.max_recursion_length:
            self.recursion_counter = 0
            raise MaxRecursionsReachedException(f"Recursion depth exceed the maximum set limit of {self.max_recursion_length}")
        self.recursion_counter += 1

    def check_event_loop(self, event: AfterModelCallEvent) -> None:
        if self.loop_counter > self.max_loop_length:
            # TODO: event.terminate is not supported on AfterModelCallEvent. Switch to a graceful
            # termination signal once the framework exposes one; for now raise to break the loop.
            e_message = f"Agent event loop exceeded the set threshold: {self.max_loop_length}. Terminating the loop"
            LOG.info(e_message)
            raise Exception(e_message)
        self.loop_counter += 1
