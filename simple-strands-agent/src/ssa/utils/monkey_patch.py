"""
Monkey patches for third-party libraries.

Import this module early to apply all patches before they are needed.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import litellm.anthropic_beta_headers_manager as beta_mgr
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.content import ContentBlock
from strands.types.event_loop import Usage
from strands.types.exceptions import ModelThrottledException
from strands.types.tools import ToolUse
import strands.event_loop.event_loop as event_loop_mod
import strands.event_loop.streaming as streaming_mod
import strands.tools.tools as tools_mod
from strands.event_loop.streaming import stream_messages as _original_stream_messages
from strands.tools.tools import (
    InvalidToolUseNameException,
    validate_tool_use_name as _original_validate_tool_use_name,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# litellm: add effort header support for bedrock providers
# ---------------------------------------------------------------------------
def _patch_bedrock_effort_header():
    config = beta_mgr._load_beta_headers_config()
    config["bedrock_converse"]["effort-2025-11-24"] = "effort-2025-11-24"
    config["bedrock"]["effort-2025-11-24"] = "effort-2025-11-24"


# ---------------------------------------------------------------------------
# strands: wrap stream_messages with per-chunk and total timeout
# ---------------------------------------------------------------------------
STREAM_TOTAL_TIMEOUT_SECONDS = 600
STREAM_CHUNK_TIMEOUT_SECONDS = 600


def _patch_stream_messages_timeout():
    """Patch strands event_loop to use a timeout-wrapped stream_messages."""

    async def stream_messages_with_timeout(
        model,
        system_prompt,
        messages,
        tool_specs,
        *,
        total_timeout: float = STREAM_TOTAL_TIMEOUT_SECONDS,
        chunk_timeout: float = STREAM_CHUNK_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> AsyncGenerator:
        start_time = asyncio.get_event_loop().time()
        stream = _original_stream_messages(model, system_prompt, messages, tool_specs, **kwargs)
        stream_iter = stream.__aiter__()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            remaining = total_timeout - elapsed

            if remaining <= 0:
                LOG.warning("stream_messages total timeout of %ss exceeded", total_timeout)
                raise ModelThrottledException(
                    f"stream_messages total timeout of {total_timeout}s exceeded"
                )

            effective_timeout = min(remaining, chunk_timeout)

            try:
                event = await asyncio.wait_for(stream_iter.__anext__(), timeout=effective_timeout)
                yield event
            except StopAsyncIteration:
                break
            except httpx.RemoteProtocolError as e:
                LOG.warning("Connection dropped during streaming: %s", e)
                raise ModelThrottledException(str(e))
            except asyncio.TimeoutError:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= total_timeout:
                    msg = f"stream_messages total timeout of {total_timeout}s exceeded"
                else:
                    msg = f"No stream event received within {chunk_timeout}s"
                LOG.warning(msg)
                raise ModelThrottledException(msg)

    event_loop_mod.stream_messages = stream_messages_with_timeout


# ---------------------------------------------------------------------------
# strands: handle_content_block_stop — append reasoning block when only a
# signature is present (no reasoning_text), so signatures aren't dropped.
# ---------------------------------------------------------------------------
def _patched_handle_content_block_stop(state: dict[str, Any]) -> dict[str, Any]:
    content: list[ContentBlock] = state["content"]

    current_tool_use = state["current_tool_use"]
    text = state["text"]
    reasoning_text = state["reasoningText"]
    citations_content = state["citationsContent"]
    redacted_content = state.get("redactedContent")

    if current_tool_use:
        if "input" not in current_tool_use:
            current_tool_use["input"] = ""

        try:
            current_tool_use["input"] = json.loads(current_tool_use["input"])
        except ValueError:
            current_tool_use["input"] = {}

        tool_use_id = current_tool_use["toolUseId"]
        tool_use_name = current_tool_use["name"]

        tool_use = ToolUse(
            toolUseId=tool_use_id,
            name=tool_use_name,
            input=current_tool_use["input"],
        )
        if "reasoningSignature" in current_tool_use:
            tool_use["reasoningSignature"] = current_tool_use["reasoningSignature"]
        content.append({"toolUse": tool_use})
        state["current_tool_use"] = {}

    elif text:
        if citations_content:
            citations_block = {"citations": citations_content, "content": [{"text": text}]}
            content.append({"citationsContent": citations_block})
            state["citationsContent"] = []
        else:
            content.append({"text": text})
        state["text"] = ""

    elif reasoning_text or "signature" in state:
        content_block: ContentBlock = {
            "reasoningContent": {
                "reasoningText": {
                    "text": state["reasoningText"],
                }
            }
        }

        if "signature" in state:
            content_block["reasoningContent"]["reasoningText"]["signature"] = state["signature"]

        content.append(content_block)
        state["reasoningText"] = ""
    elif redacted_content:
        content.append({"reasoningContent": {"redactedContent": redacted_content}})
        state["redactedContent"] = b""

    return state


def _patch_handle_content_block_stop():
    streaming_mod.handle_content_block_stop = _patched_handle_content_block_stop


# ---------------------------------------------------------------------------
# strands: accumulate reasoningTokens in EventLoopMetrics usage
# ---------------------------------------------------------------------------
_original_accumulate_usage = EventLoopMetrics._accumulate_usage


def _patched_accumulate_usage(self: EventLoopMetrics, target: Usage, source: Usage) -> None:
    _original_accumulate_usage(self, target, source)
    if "reasoningTokens" in source:
        target["reasoningTokens"] = target.get("reasoningTokens", 0) + source["reasoningTokens"]


def _patch_accumulate_usage():
    EventLoopMetrics._accumulate_usage = _patched_accumulate_usage


# ---------------------------------------------------------------------------
# strands: treat tool_use with name=None as an invalid name (upstream only
# checks for missing key, which lets None slip through and crash downstream
# `re.match` on a non-string).
# ---------------------------------------------------------------------------
def _patched_validate_tool_use_name(tool: ToolUse) -> None:
    if tool.get("name") is None:
        raise InvalidToolUseNameException("tool name missing")
    _original_validate_tool_use_name(tool)


def _patch_validate_tool_use_name():
    tools_mod.validate_tool_use_name = _patched_validate_tool_use_name
    # streaming.py did `from ..tools.tools import validate_tool_use_name`,
    # so it holds its own binding that must be updated separately.
    streaming_mod.validate_tool_use_name = _patched_validate_tool_use_name


# Apply all patches
_patch_bedrock_effort_header()
# _patch_stream_messages_timeout()
_patch_accumulate_usage()
_patch_handle_content_block_stop()
_patch_validate_tool_use_name()
