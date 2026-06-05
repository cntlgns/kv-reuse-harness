"""Anthropic Claude model provider.

- Docs: https://docs.anthropic.com/claude/reference/getting-started-with-the-api
"""

import base64
import json
import logging
import mimetypes
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, TypeVar, cast

import anthropic
import httpx
from pydantic import BaseModel
from typing_extensions import Required, Unpack, override

from strands.event_loop.streaming import process_stream
from strands.tools.structured_output.structured_output_utils import convert_pydantic_to_tool_spec
from strands.types.content import ContentBlock, Messages
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolChoiceToolDict, ToolSpec
from strands.models._validation import _has_location_source, validate_config_keys
from strands.models.model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AnthropicModel(Model):
    """Anthropic model provider implementation."""

    EVENT_TYPES = {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_stop",
    }

    OVERFLOW_MESSAGES = {
        "prompt is too long:",
        "input is too long",
        "input length exceeds context window",
        "input and output tokens exceed your context limit",
    }

    # Anthropic server-side error.type discriminators that indicate a transient /
    # retriable condition. These can be surfaced mid-stream (HTTP 200 already sent,
    # then an SSE `event: error` with a typed payload) which means they won't be
    # classified by status_code alone.
    THROTTLE_ERROR_TYPES = {"overloaded_error", "rate_limit_error", "api_error"}

    class AnthropicConfig(TypedDict, total=False):
        """Configuration options for Anthropic models.

        Attributes:
            max_tokens: Maximum number of tokens to generate.
            model_id: Calude model ID (e.g., "claude-3-7-sonnet-latest").
                For a complete list of supported models, see
                https://docs.anthropic.com/en/docs/about-claude/models/all-models.
            params: Additional model parameters (e.g., temperature).
                For a complete list of supported parameters, see https://docs.anthropic.com/en/api/messages.
        """

        max_tokens: Required[int]
        model_id: Required[str]
        params: dict[str, Any] | None

    def __init__(self, *, client_args: dict[str, Any] | None = None, **model_config: Unpack[AnthropicConfig]):
        """Initialize provider instance.

        Args:
            client_args: Arguments for the underlying Anthropic client (e.g., api_key).
                For a complete list of supported arguments, see https://docs.anthropic.com/en/api/client-sdks.
            **model_config: Configuration options for the Anthropic model.
        """
        validate_config_keys(model_config, self.AnthropicConfig)
        self.config = AnthropicModel.AnthropicConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        self.client = anthropic.AsyncAnthropic(**client_args)

    @override
    def update_config(self, **model_config: Unpack[AnthropicConfig]) -> None:  # type: ignore[override]
        """Update the Anthropic model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.AnthropicConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> AnthropicConfig:
        """Get the Anthropic model configuration.

        Returns:
            The Anthropic model configuration.
        """
        return self.config

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any]:
        """Format an Anthropic content block.

        Args:
            content: Message content.

        Returns:
            Anthropic formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to an Anthropic-compatible format.
        """
        if "document" in content:
            mime_type = mimetypes.types_map.get(f".{content['document']['format']}", "application/octet-stream")
            return {
                "source": {
                    "data": (
                        content["document"]["source"]["bytes"].decode("utf-8")
                        if mime_type == "text/plain"
                        else base64.b64encode(content["document"]["source"]["bytes"]).decode("utf-8")
                    ),
                    "media_type": mime_type,
                    "type": "text" if mime_type == "text/plain" else "base64",
                },
                "title": content["document"]["name"],
                "type": "document",
            }

        if "image" in content:
            return {
                "source": {
                    "data": base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8"),
                    "media_type": mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream"),
                    "type": "base64",
                },
                "type": "image",
            }

        if "reasoningContent" in content:
            return {
                "signature": content["reasoningContent"]["reasoningText"]["signature"],
                "thinking": content["reasoningContent"]["reasoningText"]["text"],
                "type": "thinking",
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        if "toolUse" in content:
            return {
                "id": content["toolUse"]["toolUseId"],
                "input": content["toolUse"]["input"],
                "name": content["toolUse"]["name"],
                "type": "tool_use",
            }

        if "toolResult" in content:
            return {
                "content": [
                    self._format_request_message_content(
                        {"text": json.dumps(tool_result_content["json"])}
                        if "json" in tool_result_content
                        else cast(ContentBlock, tool_result_content)
                    )
                    for tool_result_content in content["toolResult"]["content"]
                ],
                "is_error": content["toolResult"]["status"] == "error",
                "tool_use_id": content["toolResult"]["toolUseId"],
                "type": "tool_result",
            }

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_request_messages(self, messages: Messages) -> list[dict[str, Any]]:
        """Format an Anthropic messages array.

        Args:
            messages: List of message objects to be processed by the model.

        Returns:
            An Anthropic messages array.
        """
        formatted_messages = []

        for message in messages:
            formatted_contents: list[dict[str, Any]] = []

            for content in message["content"]:
                if "cachePoint" in content:
                    formatted_contents[-1]["cache_control"] = {"type": "ephemeral"}
                    continue

                # Check for location sources in image, document, or video content
                if _has_location_source(content):
                    logger.warning("Location sources are not supported by Anthropic | skipping content block")
                    continue

                formatted_contents.append(self._format_request_message_content(content))

            if formatted_contents:
                formatted_messages.append({"content": formatted_contents, "role": message["role"]})

        return formatted_messages

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format an Anthropic streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.

        Returns:
            An Anthropic streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to an Anthropic-compatible
                format.
        """
        return {
            "max_tokens": self.config["max_tokens"],
            "messages": self._format_request_messages(messages),
            "model": self.config["model_id"],
            "tools": [
                {
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "input_schema": tool_spec["inputSchema"]["json"],
                }
                for tool_spec in tool_specs or []
            ],
            **(self._format_tool_choice(tool_choice)),
            **({"system": system_prompt} if system_prompt else {}),
            **(self.config.get("params") or {}),
        }

    @staticmethod
    def _extract_error_type(error: anthropic.APIStatusError) -> str | None:
        """Extract the Anthropic `error.type` discriminator from an APIStatusError body.

        The Anthropic API reports typed errors as `{"type": "error", "error": {"type": "<kind>", ...}}`.
        Mid-stream errors (e.g. `overloaded_error`) arrive after a 200 status, so the SDK can
        only classify them via the body payload — not via HTTP status code.
        """
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict):
                err_type = inner.get("type")
                if isinstance(err_type, str):
                    return err_type
        return None

    @staticmethod
    def _format_tool_choice(tool_choice: ToolChoice | None) -> dict:
        if tool_choice is None:
            return {}

        if "any" in tool_choice:
            return {"tool_choice": {"type": "any"}}
        elif "auto" in tool_choice:
            return {"tool_choice": {"type": "auto"}}
        elif "tool" in tool_choice:
            return {"tool_choice": {"type": "tool", "name": cast(ToolChoiceToolDict, tool_choice)["tool"]["name"]}}
        else:
            return {}

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the Anthropic response events into standardized message chunks.

        Args:
            event: A response event from the Anthropic model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as we control chunk_type in the stream method.
        """
        match event["type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_block_start":
                content = event["content_block"]

                if content["type"] == "tool_use":
                    return {
                        "contentBlockStart": {
                            "contentBlockIndex": event["index"],
                            "start": {
                                "toolUse": {
                                    "name": content["name"],
                                    "toolUseId": content["id"],
                                }
                            },
                        }
                    }

                return {"contentBlockStart": {"contentBlockIndex": event["index"], "start": {}}}

            case "content_block_delta":
                delta = event["delta"]

                match delta["type"]:
                    case "signature_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "reasoningContent": {
                                        "signature": delta["signature"],
                                    },
                                },
                            },
                        }

                    case "thinking_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "reasoningContent": {
                                        "text": delta["thinking"],
                                    },
                                },
                            },
                        }

                    case "input_json_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "toolUse": {
                                        "input": delta["partial_json"],
                                    },
                                },
                            },
                        }

                    case "text_delta":
                        return {
                            "contentBlockDelta": {
                                "contentBlockIndex": event["index"],
                                "delta": {
                                    "text": delta["text"],
                                },
                            },
                        }

                    case _:
                        raise RuntimeError(
                            f"event_type=<content_block_delta>, delta_type=<{delta['type']}> | unknown type"
                        )

            case "content_block_stop":
                return {"contentBlockStop": {"contentBlockIndex": event["index"]}}

            case "message_stop":
                message = event["message"]

                return {"messageStop": {"stopReason": message["stop_reason"]}}

            case "metadata":
                usage = event["usage"]

                usage_data: dict[str, Any] = {
                    "inputTokens": usage["input_tokens"],
                    "outputTokens": usage["output_tokens"],
                    "totalTokens": usage["input_tokens"] + usage["output_tokens"],
                }
                if (cache_read := usage.get("cache_read_input_tokens")) is not None:
                    usage_data["cacheReadInputTokens"] = cache_read
                if (cache_write := usage.get("cache_creation_input_tokens")) is not None:
                    usage_data["cacheWriteInputTokens"] = cache_write

                return {
                    "metadata": {
                        "usage": usage_data,
                        "metrics": {
                            "latencyMs": 0,  # TODO
                        },
                    }
                }

            case _:
                raise RuntimeError(f"event_type=<{event['type']} | unknown type")

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
        """Stream conversation with the Anthropic model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled, times out, fails to connect,
                or the Anthropic service returns a transient server / overload error. This also
                covers mid-stream SSE `event: error` payloads such as
                `{"type": "error", "error": {"type": "overloaded_error", ...}}` which the SDK
                raises as an `APIStatusError` with status_code 200 (since the 2xx response
                header already flushed before the error arrived).
            anthropic.AuthenticationError: If the API key is invalid or missing.
            anthropic.PermissionDeniedError: If the caller is not permitted to use the resource.
            anthropic.NotFoundError: If the requested model or resource does not exist.
            anthropic.BadRequestError: For non-context-overflow 400 errors (caller bug — retrying won't help).
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        try:
            async with self.client.messages.stream(**request) as stream:
                logger.debug("got response from model")
                async for event in stream:
                    if event.type in AnthropicModel.EVENT_TYPES:
                        if event.type == "message_stop":
                            # Build dict directly to avoid Pydantic serialization warnings
                            # when the message contains ParsedTextBlock objects (issue #1746)
                            yield self.format_chunk(
                                {
                                    "type": "message_stop",
                                    "message": {"stop_reason": event.message.stop_reason},
                                }
                            )
                        else:
                            yield self.format_chunk(event.model_dump())

                try:
                    message_snapshot = await stream.get_final_message()
                except AssertionError as e:
                    logger.warning("error=<%s> | failed to retrieve message snapshot, usage metadata unavailable", e)
                else:
                    yield self.format_chunk({"type": "metadata", "usage": message_snapshot.usage.model_dump()})

        except anthropic.RateLimitError as error:
            logger.warning("anthropic rate limit error: %s", error)
            raise ModelThrottledException(str(error)) from error

        except anthropic.BadRequestError as error:
            if any(overflow_message in str(error).lower() for overflow_message in AnthropicModel.OVERFLOW_MESSAGES):
                logger.warning("anthropic context window overflow error detected")
                raise ContextWindowOverflowException(str(error)) from error

            logger.error("anthropic bad request error: %s", error)
            raise ModelThrottledException(str(error)) from error

        except anthropic.APITimeoutError as error:
            logger.warning("anthropic request timed out: %s", error)
            raise ModelThrottledException(str(error)) from error

        except anthropic.APIConnectionError as error:
            logger.warning("anthropic connection error: %s", error)
            raise ModelThrottledException(str(error)) from error

        except anthropic.InternalServerError as error:
            # Covers 5xx incl. 529 "Overloaded" — safe to retry as throttle.
            logger.warning("anthropic internal server error: %s", error)
            raise ModelThrottledException(str(error)) from error

        except anthropic.APIStatusError as error:
            # Catch-all for other HTTP status errors and mid-stream SSE `event: error`
            # payloads. The latter carry status_code == 200 (the stream already opened) so
            # classification must use the body's `error.type` discriminator.
            status_code = getattr(error, "status_code", None)
            error_type = self._extract_error_type(error)

            if error_type in AnthropicModel.THROTTLE_ERROR_TYPES or status_code == 529:
                logger.warning(
                    "anthropic transient error (status=%s, error_type=%s): %s",
                    status_code,
                    error_type,
                    error,
                )
                raise ModelThrottledException(str(error)) from error

            error_message = str(error)
            if any(overflow_message in error_message.lower() for overflow_message in AnthropicModel.OVERFLOW_MESSAGES):
                logger.warning("anthropic context window overflow error detected via APIStatusError")
                raise ContextWindowOverflowException(error_message) from error

            # Non-retriable 4xx (e.g. 409 Conflict, 422 Unprocessable) — surface as-is.
            if isinstance(status_code, int) and 400 <= status_code < 500:
                logger.error("anthropic api status error (status=%s): %s", status_code, error)
                raise

            logger.warning("anthropic api status error (status=%s): %s", status_code, error)
            raise ModelThrottledException(error_message) from error

        except anthropic.APIError as error:
            # Residual SDK errors that aren't APIStatusError (e.g. response validation).
            logger.warning("anthropic api error: %s", error)
            raise ModelThrottledException(str(error)) from error

        except httpx.HTTPError as error:
            # Transport-level errors raised mid-stream (e.g. RemoteProtocolError, ReadTimeout)
            # can escape the Anthropic SDK's wrapping.
            logger.warning("httpx transport error during stream: %s", error)
            raise ModelThrottledException(str(error)) from error

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        tool_spec = convert_pydantic_to_tool_spec(output_model)

        response = self.stream(
            messages=prompt,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
            tool_choice=cast(ToolChoice, {"any": {}}),
            **kwargs,
        )
        async for event in process_stream(response):
            yield event

        stop_reason, messages, _, _ = event["stop"]

        if stop_reason != "tool_use":
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "tool_use".')

        content = messages["content"]
        output_response: dict[str, Any] | None = None
        for block in content:
            # if the tool use name doesn't match the tool spec name, skip, and if the block is not a tool use, skip.
            # if the tool use name never matches, raise an error.
            if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
                output_response = block["toolUse"]["input"]
            else:
                continue

        if output_response is None:
            raise ValueError("No valid tool use or tool use input was found in the Anthropic response.")

        yield {"output": output_model(**output_response)}