"""xAI Grok model provider using the xai_sdk (gRPC).

Implements the strands Model interface by wrapping xai_sdk's streaming chat API.
Supports text, reasoning content, tool calling, and prompt caching via x-grok-conv-id.
"""

import json
import logging
import secrets
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, TypeVar

import pydantic
from typing_extensions import Required, Unpack, override
from xai_sdk import AsyncClient
from xai_sdk.chat import system as xai_system, tool as xai_tool, tool_result as xai_tool_result, user as xai_user

from strands.models.model import Model
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

LOG = logging.getLogger(__name__)

T = TypeVar("T", bound=pydantic.BaseModel)


class XAIModel(Model):
    """xAI Grok model provider using the native xai_sdk.

    Uses gRPC streaming via xai_sdk.AsyncClient for efficient token delivery.
    Prompt caching is automatic; set conv_id to maximize cache hits by routing
    requests to the same server.
    """

    class XAIConfig(TypedDict, total=False):
        """Configuration for xAI models.

        Attributes:
            model_id: Grok model ID (e.g. "grok-4.20-reasoning").
            params: Additional model parameters passed to chat.create().
            conv_id: Conversation ID for prompt cache affinity (x-grok-conv-id).
            timeout: Request timeout in seconds (default: 3600).
        """

        model_id: Required[str]
        params: dict[str, Any]
        conv_id: str | None
        timeout: float

    def __init__(self, use_previous_id: bool = True, **model_config: Unpack[XAIConfig]) -> None:
        self.config = XAIModel.XAIConfig(**model_config)
        # Auto-generate a conv_id for prompt cache affinity if not provided.
        # This ensures all stream() calls in a multi-turn agent loop hit the
        # same server, maximizing cache hits across tool-use rounds.
        if not self.config.get("conv_id"):
            self.config["conv_id"] = f"ssa_{secrets.token_urlsafe(12)}"
        self.use_previous_id = use_previous_id
        self._previous_response_id: str | None = None
        LOG.debug("config=<%s> | initializing xAI model", self.config)

    @override
    def update_config(self, **model_config: Unpack[XAIConfig]) -> None:  # type: ignore[override]
        self.config.update(model_config)

    @override
    def get_config(self) -> XAIConfig:
        return self.config

    def _get_client(self) -> AsyncClient:
        """Create an AsyncClient for this request.

        A new client is created per-request to avoid sharing gRPC channels across
        asyncio event loops. API key is read from XAI_API_KEY env var (xai_sdk default).
        """
        kwargs: dict[str, Any] = {
            "timeout": self.config.get("timeout", 3600),
        }

        # Route to same server for prompt cache hits
        if conv_id := self.config.get("conv_id"):
            kwargs["metadata"] = (("x-grok-conv-id", conv_id),)

        return AsyncClient(**kwargs)

    # ── Message formatting: strands Messages → xai_sdk messages ──

    @classmethod
    def _format_messages(cls, messages: Messages, system_prompt: str | None = None) -> list[Any]:
        """Convert strands Messages to xai_sdk message objects.

        Handles text, toolUse, and toolResult content blocks.
        System prompt is prepended as a system message.
        """
        formatted = []

        if system_prompt:
            formatted.append(xai_system(system_prompt))

        for message in messages:
            role = message["role"]
            contents = message["content"]

            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            tool_results: list[tuple[str, str | None]] = []

            for content in contents:
                if "text" in content:
                    text_parts.append(content["text"])
                elif "toolUse" in content:
                    tu = content["toolUse"]
                    tool_uses.append({
                        "name": tu["name"],
                        "toolUseId": tu["toolUseId"],
                        "input": tu["input"],
                    })
                elif "toolResult" in content:
                    tr = content["toolResult"]
                    output_parts = []
                    for c in tr.get("content", []):
                        if "text" in c:
                            output_parts.append(c["text"])
                        elif "json" in c:
                            output_parts.append(json.dumps(c["json"]))
                    result_str = "\n".join(output_parts) if output_parts else ""
                    tool_results.append((result_str, tr.get("toolUseId")))

            # Emit user/assistant text messages
            if text_parts:
                combined = "\n".join(text_parts)
                if role == "user":
                    formatted.append(xai_user(combined))
                # Assistant text is part of the response history; xai_sdk manages this
                # via chat.append(response) in the agentic loop, but for initial context
                # we use the assistant helper
                elif role == "assistant":
                    from xai_sdk.chat import assistant as xai_assistant
                    formatted.append(xai_assistant(combined))

            # Emit tool results
            for result_str, tool_call_id in tool_results:
                formatted.append(xai_tool_result(result_str, tool_call_id=tool_call_id))

        return formatted

    @classmethod
    def _format_tools(cls, tool_specs: list[ToolSpec] | None) -> list[Any]:
        """Convert strands ToolSpecs to xai_sdk tool definitions."""
        if not tool_specs:
            return []
        return [
            xai_tool(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["inputSchema"]["json"],
            )
            for spec in tool_specs
        ]

    # ── Stream event formatting: xai_sdk chunks → strands StreamEvents ──

    @classmethod
    def _format_chunk(cls, event: dict[str, Any]) -> StreamEvent:
        """Convert internal event dict to strands StreamEvent."""
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event.get("data_type") == "tool":
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": event["name"],
                                    "toolUseId": event["toolUseId"],
                                },
                            },
                        },
                    }
                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                match event.get("data_type"):
                    case "tool":
                        return {
                            "contentBlockDelta": {
                                "delta": {"toolUse": {"input": event["data"]}}
                            }
                        }
                    case "reasoning":
                        return {
                            "contentBlockDelta": {
                                "delta": {"reasoningContent": {"text": event["data"]}}
                            }
                        }
                    case _:
                        return {
                            "contentBlockDelta": {
                                "delta": {"text": event["data"]}
                            }
                        }

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                return {"messageStop": {"stopReason": event.get("stop_reason", "end_turn")}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": event["usage"],
                        "metrics": {"latencyMs": 0},
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    # ── Core streaming implementation ──

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the xAI Grok model.

        Converts strands messages/tools to xai_sdk format, streams the response,
        and yields strands StreamEvents for text, reasoning, and tool call blocks.
        """
        LOG.debug("formatting xAI request")

        xai_messages = self._format_messages(messages, system_prompt)
        xai_tools = self._format_tools(tool_specs)

        params = dict(self.config.get("params", {}))

        async with self._get_client() as client:
            try:
                # When chaining via previous_response_id, server holds prior context;
                # only send the latest messages.
                if self._previous_response_id:
                    chain_messages = xai_messages[-1:] if xai_messages else xai_messages
                else:
                    chain_messages = xai_messages

                create_kwargs: dict[str, Any] = {
                    "model": self.config["model_id"],
                    "messages": chain_messages,
                    **(
                        {"store_messages": True} if self.use_previous_id else {}
                    ),
                }
                if self._previous_response_id:
                    create_kwargs["previous_response_id"] = self._previous_response_id
                if xai_tools:
                    create_kwargs["tools"] = xai_tools
                create_kwargs.update(params)

                chat = client.chat.create(**create_kwargs)

                LOG.debug("streaming xAI response")
                yield self._format_chunk({"chunk_type": "message_start"})

                data_type: str | None = None  # tracks current content block type
                tool_calls_seen: list[dict[str, Any]] = []
                response = None

                async for response, chunk in chat.stream():
                    # ── Reasoning content ──
                    if chunk.reasoning_content:
                        if data_type != "reasoning":
                            if data_type is not None:
                                yield self._format_chunk({"chunk_type": "content_stop"})
                            yield self._format_chunk({"chunk_type": "content_start", "data_type": "reasoning"})
                            data_type = "reasoning"
                        yield self._format_chunk({
                            "chunk_type": "content_delta",
                            "data_type": "reasoning",
                            "data": chunk.reasoning_content,
                        })

                    # ── Text content ──
                    if chunk.content:
                        if data_type != "text":
                            if data_type is not None:
                                yield self._format_chunk({"chunk_type": "content_stop"})
                            yield self._format_chunk({"chunk_type": "content_start", "data_type": "text"})
                            data_type = "text"
                        yield self._format_chunk({
                            "chunk_type": "content_delta",
                            "data_type": "text",
                            "data": chunk.content,
                        })

                    # ── Tool calls (delivered whole in a single chunk) ──
                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            tool_call_id = getattr(tc, "id", None) or f"tooluse_{secrets.token_urlsafe(16)}"
                            func_name = tc.function.name
                            func_args = tc.function.arguments

                            tool_calls_seen.append({
                                "id": tool_call_id,
                                "name": func_name,
                                "arguments": func_args,
                            })

                # Close any open content block
                if data_type is not None:
                    yield self._format_chunk({"chunk_type": "content_stop"})

                # Emit tool call blocks
                for tc_data in tool_calls_seen:
                    yield self._format_chunk({
                        "chunk_type": "content_start",
                        "data_type": "tool",
                        "name": tc_data["name"],
                        "toolUseId": tc_data["id"],
                    })
                    yield self._format_chunk({
                        "chunk_type": "content_delta",
                        "data_type": "tool",
                        "data": tc_data["arguments"],
                    })
                    yield self._format_chunk({"chunk_type": "content_stop"})

                # Capture response ID for chaining subsequent requests
                self._previous_response_id = getattr(response, "id", None) if response and self.use_previous_id else None

                # Stop reason
                stop_reason = "end_turn"
                if tool_calls_seen:
                    stop_reason = "tool_use"
                elif response and getattr(response, "finish_reason", None) == "length":
                    stop_reason = "max_tokens"

                yield self._format_chunk({"chunk_type": "message_stop", "stop_reason": stop_reason})

                # Usage metadata
                if response and response.usage:
                    usage_dict: dict[str, Any] = {
                        "inputTokens": getattr(response.usage, "prompt_tokens", 0),
                        "outputTokens": getattr(response.usage, "completion_tokens", 0),
                        "totalTokens": getattr(response.usage, "total_tokens", 0),
                    }
                    # Reasoning tokens
                    if hasattr(response.usage, "reasoning_tokens") and response.usage.reasoning_tokens:
                        usage_dict["reasoningTokens"] = response.usage.reasoning_tokens
                    # Cached prompt tokens
                    if hasattr(response.usage, "cached_prompt_text_tokens"):
                        usage_dict["cacheReadInputTokens"] = response.usage.cached_prompt_text_tokens

                    yield self._format_chunk({"chunk_type": "metadata", "usage": usage_dict})

            except Exception as e:
                err_message = str(e)
                # gRPC errors surface as generic exceptions; check for common patterns
                err_lower = err_message.lower()
                if "context" in err_lower and ("length" in err_lower or "overflow" in err_lower or "exceed" in err_lower):
                    LOG.warning("xAI context window overflow: %s", err_message)
                    raise ContextWindowOverflowException(err_message) from e
                if any(kw in err_lower for kw in ["rate limit", "throttl", "resource_exhausted", "unavailable", "timeout", "connection"]):
                    LOG.warning("xAI transient error, treating as throttled: %s", err_message)
                    raise ModelThrottledException(err_message) from e
                LOG.warning("xAI error: %s", err_message)
                raise ModelThrottledException(err_message) from e

        LOG.debug("finished streaming xAI response")

    @override
    async def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Structured output is not natively supported by xai_sdk; fall back to JSON mode."""
        schema = output_model.model_json_schema()
        augmented_prompt = (
            f"{system_prompt or ''}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema)}"
        ).strip()

        xai_messages = self._format_messages(prompt, augmented_prompt)

        async with self._get_client() as client:
            chat = client.chat.create(
                model=self.config["model_id"],
                messages=xai_messages,
                **dict(self.config.get("params", {})),
            )
            response = await chat.sample()
            parsed = output_model.model_validate_json(response.content)
            yield {"output": parsed}
