import logging
import litellm
from collections.abc import AsyncGenerator
from strands.types.event_loop import Usage
from strands.types.exceptions import ModelThrottledException, ContextWindowOverflowException
from strands.types.streaming import MetadataEvent, StreamEvent
from strands.types.content import Messages, SystemContentBlock
from strands.models.litellm import LiteLLMModel
from typing import Any
from typing_extensions import override
from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    BadGatewayError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from ssa.models.openai import SROpenAIModel


LOG = logging.getLogger(__name__)


class SRLiteLLMModel(LiteLLMModel):
    @override
    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        """Format a LiteLLM response event into a standardized message chunk.

        This method overrides OpenAI's format_chunk to handle the metadata case
        with prompt caching support. All other chunk types use the parent implementation.

        It further overwrite the format_chunk to access signature field in reasoning
        content which is needed by anthropic models

        Args:
            event: A response event from the LiteLLM model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
        """
        # Handle metadata case with prompt caching support
        if event["chunk_type"] == "metadata":
            usage_data: Usage = {
                "inputTokens": event["data"].prompt_tokens,
                "outputTokens": event["data"].completion_tokens,
                "totalTokens": event["data"].total_tokens,
            }

            # Only LiteLLM over Anthropic supports cache write tokens
            # Waiting until a more general approach is available to set cacheWriteInputTokens
            if tokens_details := getattr(event["data"], "prompt_tokens_details", None):
                if cached := getattr(tokens_details, "cached_tokens", None):
                    usage_data["cacheReadInputTokens"] = cached
            if creation := getattr(event["data"], "cache_creation_input_tokens", None):
                usage_data["cacheWriteInputTokens"] = creation

            return StreamEvent(
                metadata=MetadataEvent(
                    metrics={
                        "latencyMs": 0,  # TODO
                    },
                    usage=usage_data,
                )
            )
        # Handle reasoning content signature
        if event["chunk_type"] == "content_delta":
            if event["data_type"] == "reasoning_content":
                if event.get("signature"):
                    return {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": event["signature"]}}}}
        # For all other cases, use the parent implementation
        return super().format_chunk(event, **kwargs)

    @override
    @classmethod
    def format_request_messages(
        cls,
        messages: Messages,
        system_prompt: str | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format a LiteLLM compatible messages array with cache point support.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model (for legacy compatibility).
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A LiteLLM compatible messages array.
        """
        formatted_messages = cls._format_system_messages(system_prompt, system_prompt_content=system_prompt_content)
        formatted_messages.extend(SROpenAIModel._format_regular_messages(messages))

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    @override
    async def _handle_streaming_response(self, litellm_request: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        """
        Gemini models:
            Returns stop_reason as "stop" for successful tool-call. To prevent misinterpretation as stop-loop
            We overwrite stop_reason for non-zero tool-calls message as "tool_calls"
        
        Wrap parent streaming response with retry logic for ContextOverFlowErorr, RateLimitError, ServiceUnavailableError.
        """
        try:
            # async for chunk in super()._handle_streaming_response(litellm_request):
            #     yield chunk
            # return
            response = await litellm.acompletion(**self.client_args, **litellm_request)

            yield self.format_chunk({"chunk_type": "message_start"})

            tool_calls: dict[int, list[Any]] = {}
            data_type: str | None = None
            finish_reason: str | None = None

            async for event in response:
                # Defensive: skip events with empty or missing choices
                if not getattr(event, "choices", None):
                    continue
                choice = event.choices[0]

                # Process content using shared logic
                async for updated_data_type, chunk in self._process_choice_content(
                    choice, data_type, tool_calls, is_streaming=True
                ):
                    data_type = updated_data_type
                    yield chunk

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    if data_type:
                        yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                    break

            if len(tool_calls) > 0:
                # Gemini models return "stop" for tool-calls. Overwrite here to make a tool-call intent instead for autonomous setting
                finish_reason = "tool_calls" 
            # Process tool calls
            async for chunk in self._process_tool_calls(tool_calls):
                yield chunk

            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

            # Skip remaining events as we don't have use for anything except the final usage payload
            async for event in response:
                _ = event
                if usage := getattr(event, "usage", None):
                    yield self.format_chunk({"chunk_type": "metadata", "data": usage})

        except ContextWindowExceededError as e:
            LOG.warning(f"Received context window overflow error. Retry with conversation maanger. Details: {e}")
            raise ContextWindowOverflowException from e
        except RateLimitError as e:
            err_message = str(e)
            LOG.warning(f"rate limit exceeded, retrying. Details: {err_message}")
            raise ModelThrottledException(message=err_message) from e
        except (APIConnectionError, BadGatewayError, BadRequestError, InternalServerError, ServiceUnavailableError) as e:
            err_message = str(e)
            LOG.warning(f"service unavailable temporarily. Treating this as possible throttling and retying. Details: {e}")
            raise ModelThrottledException(message=err_message) from e
        except Timeout as e:
            err_message = str(e)
            LOG.warning(f"Timeout from service. Treating this as possible throttling and retying. Details: {e}")
            raise ModelThrottledException(message=err_message) from e

    @override
    async def _handle_non_streaming_response(self, litellm_request: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        """Wrap parent non-streaming response with retry logic for transient errors."""
        try:
            async for chunk in super()._handle_non_streaming_response(litellm_request):
                yield chunk
        except ContextWindowExceededError as e:
            LOG.warning(f"Received context window overflow error. Retry with conversation manager. Details: {e}")
            raise ContextWindowOverflowException from e
        except RateLimitError as e:
            err_message = str(e)
            LOG.warning(f"rate limit exceeded, retrying. Details: {err_message}")
            raise ModelThrottledException(message=err_message) from e
        except (BadRequestError, ServiceUnavailableError, InternalServerError) as e:
            err_message = str(e)
            LOG.warning(f"service unavailable temporarily. Treating this as possible throttling and retrying. Details: {e}")
            raise ModelThrottledException(message=err_message) from e
        except Timeout as e:
            err_message = str(e)
            LOG.warning(f"Timeout from service. Treating this as possible throttling and retrying. Details: {e}")
            raise ModelThrottledException(message=err_message) from e

    @override
    async def _process_choice_content(
        self, choice: Any, data_type: str | None, tool_calls: dict[int, list[Any]], is_streaming: bool = True
    ) -> AsyncGenerator[tuple[str | None, StreamEvent], None]:
        """Process content from a choice object (streaming or non-streaming).
        It overrides the parent implentation to emit signature from reasoning content blocks
        This is needed for interleaved reasoning in anthropic models 

        Args:
            choice: The choice object from the response.
            data_type: Current data type being processed.
            tool_calls: Dictionary to collect tool calls.
            is_streaming: Whether this is from a streaming response.

        Yields:
            Tuples of (updated_data_type, stream_event).
        """
        # Get the content source - this is the only difference between streaming/non-streaming
        # We use duck typing here: both choice.delta and choice.message have the same interface
        # (reasoning_content, content, tool_calls attributes) but different object structures
        content_source = choice.delta if is_streaming else choice.message

        # Process reasoning content
        if hasattr(content_source, "reasoning_content") and content_source.reasoning_content:
            chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
            for chunk in chunks:
                yield data_type, chunk
            chunk = self.format_chunk(
                {
                    "chunk_type": "content_delta",
                    "data_type": "reasoning_content",
                    "data": content_source.reasoning_content,
                }
            )
            yield data_type, chunk

        # Process reasoning content signature
        if (
            hasattr(content_source, "reasoning_content") and
            hasattr(content_source, "thinking_blocks") and
            isinstance(content_source.thinking_blocks, list)
        ):
            for tb in content_source.thinking_blocks:
                if isinstance(tb, dict) and tb.get("type", "") == "thinking":
                    signature = tb.get("signature")
                    if signature:
                        chunk = self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "reasoning_content",
                                "data": "",
                                "signature": signature,
                            }
                        )
                        yield data_type, chunk
        

        # Process text content
        if hasattr(content_source, "content") and content_source.content:
            chunks, data_type = self._stream_switch_content("text", data_type)
            for chunk in chunks:
                yield data_type, chunk
            chunk = self.format_chunk(
                {
                    "chunk_type": "content_delta",
                    "data_type": "text",
                    "data": content_source.content,
                }
            )
            yield data_type, chunk

        # Process tool calls
        if hasattr(content_source, "tool_calls") and content_source.tool_calls:
            if is_streaming:
                # Streaming: tool calls have index attribute for out-of-order delivery
                for tool_call in content_source.tool_calls:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)
            else:
                # Non-streaming: tool calls arrive in order, use enumerated index
                for i, tool_call in enumerate(content_source.tool_calls):
                    tool_calls.setdefault(i, []).append(tool_call)

