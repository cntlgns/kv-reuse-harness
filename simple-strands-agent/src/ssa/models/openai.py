
import logging
import os
import time
import uuid
import httpx
import openai
import json
from collections.abc import AsyncGenerator
from strands.types.exceptions import ModelThrottledException, ContextWindowOverflowException
from strands.types.streaming import StreamEvent
from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.tools import ToolChoice, ToolSpec
from strands.models.openai import OpenAIModel
from typing import Any, cast
from typing_extensions import override
from strands.models.openai import _CONTEXT_OVERFLOW_MESSAGES


LOG = logging.getLogger(__name__)


class HarmonyParseException(Exception):
    """gpt-oss harmony parser emitted a retryable parse error mid-stream.

    Surfaced from `stream()` so a HookProvider listening on AfterModelCallEvent
    can set `event.retry = True` to trigger an immediate model retry — without
    going through strands' ModelThrottledException backoff path.
    """


_HARMONY_ENC: Any = None


def _decode_harmony(token_ids: list[int]) -> list[str]:
    """Decode harmony token ids one-by-one so special tokens stay visible."""
    global _HARMONY_ENC
    if _HARMONY_ENC is None:
        try:
            import tiktoken
            _HARMONY_ENC = tiktoken.get_encoding("o200k_harmony")
        except Exception as e:
            LOG.debug("harmony encoder unavailable: %s", e)
            _HARMONY_ENC = False
    if not _HARMONY_ENC:
        return []
    return [_HARMONY_ENC.decode([tid]) for tid in token_ids]

# Extends strands' _CONTEXT_OVERFLOW_MESSAGES
_EXPANDED_CONTEXT_OVERFLOW_MESSAGES = (
    *_CONTEXT_OVERFLOW_MESSAGES,
    "maximum context length",
    "context_length_exceeded",
    "reduce the length of the input prompt",
    "prompt is too long",
)


class SROpenAIModel(OpenAIModel):

    def __init__(
        self,
        use_previous_id: bool = False,
        use_responses_api: bool = True,
        refresh_bedrock_token: bool = False,
        refresh_gcloud_token: bool = False,
        include_reasoning_in_history: bool = False,
        cache_client: bool = True,
        provide_session_id: bool | str = False,
        request_log: bool = False,
        **kwargs: Any,
    ) -> None:
        self.use_previous_id = use_previous_id
        self.use_responses_api = use_responses_api
        self.refresh_bedrock_token = refresh_bedrock_token
        self.refresh_gcloud_token = refresh_gcloud_token
        self.include_reasoning_in_history = include_reasoning_in_history
        self.cache_client = cache_client
        if isinstance(provide_session_id, str):
            self.session_id: str | None = provide_session_id
        elif provide_session_id:
            self.session_id = uuid.uuid4().hex
        else:
            self.session_id = None
        if self.session_id:
            LOG.info("Using X-Session-Id: %s", self.session_id)
        self._previous_response_id: str | None = None
        self._reasoning_field_name: str = "reasoning_content"
        # Last gateway error code/message surfaced to downstream hooks
        # (e.g. content_hook reads `invalid_prompt` to trigger truncate-and-retry).
        self._last_error_code: str | None = None
        self._last_error_message: str | None = None
        # Harmony token ids accumulated during the most recent chat-completions
        self._last_stream_token_ids: list[int] = []
        # Per-request telemetry (requests.jsonl + request_id tagging), off by
        # default so measurement runs opt in without touching other configs.
        self.request_log = request_log
        self._call_seq = 0
        # Distinguishes repeated runs of the same (bench, arm, task) triple so
        # request_ids stay globally unique in the server-side ledger.
        self._run_token = uuid.uuid4().hex[:6]
        self._current_request_id: str | None = None
        super().__init__(**kwargs)
        if self.cache_client:
            # Build once and reuse
            self._custom_client = openai.AsyncOpenAI(**self.client_args)

    def _refresh_bedrock_token(self) -> None:
        from aws_bedrock_token_generator import provide_token
        new_token = provide_token()
        self.client_args["api_key"] = new_token
        if self.cache_client:
            self._custom_client = openai.AsyncOpenAI(**self.client_args)

    def _refresh_gcloud_token(self) -> None:
        import subprocess
        gcloud = os.environ.get("GCLOUD_BIN", "gcloud")
        new_token = subprocess.check_output(
            [gcloud, "auth", "print-access-token"], text=True
        ).strip()
        self.client_args["api_key"] = new_token
        if self.cache_client:
            self._custom_client = openai.AsyncOpenAI(**self.client_args)

    @classmethod
    def format_request_message_content(cls, content: ContentBlock) -> dict[str, Any]:
        """Format a LiteLLM content block.

        Args:
            content: Message content.

        Returns:
            LiteLLM formatted content block.

        Raises:
            TypeError: If the content block type cannot be converted to a LiteLLM-compatible format.
        """
        if "reasoningContent" in content:
            return {
                "signature": content["reasoningContent"]["reasoningText"].get("signature", "default_signature"),
                "thinking": content["reasoningContent"]["reasoningText"]["text"],
                "type": "thinking",
            }

        if "video" in content:
            return {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": content["video"]["source"]["bytes"],
                },
            }

        return super().format_request_message_content(content)

    @classmethod
    def _format_regular_messages(cls, messages: Messages, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Override:
        Suppress warning of reasoning content not supported
        """
        formatted_messages: list[dict[str, Any]] = []

        for message in messages:
            contents = message["content"]

            # Support reasoning_block in the assistant message for subsequent conversations
            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse",])
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.extend(formatted_tool_messages)
            formatted_messages.append(formatted_message)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    @classmethod
    def _convert_messages_to_input(
        cls,
        messages: Messages,
        system_prompt: str | None = None,
        chain: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert strands Messages to Responses API input items.

        The Responses API expects a flat list of input items:
          - {"role": "user", "content": "..."}
          - {"role": "assistant", "content": "..."}
          - {"type": "function_call", "name": ..., "arguments": ..., "call_id": ...}
          - {"type": "function_call_output", "call_id": ..., "output": ...}
        """
        items: list[dict[str, Any]] = []

        # If chaining via previous_response_id, the server holds prior context
        messages_to_send = messages[-1:] if chain else messages

        for message in messages_to_send:
            role = message["role"]
            contents = message["content"]

            # Collect text parts, tool uses, and tool results from this message
            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for content in contents:
                if "text" in content:
                    text_parts.append(content["text"])
                elif "toolUse" in content:
                    tu = content["toolUse"]
                    tool_uses.append({
                        "type": "function_call",
                        "name": tu["name"],
                        "arguments": json.dumps(tu["input"]) if isinstance(tu["input"], dict) else tu["input"],
                        "call_id": tu["toolUseId"],
                    })
                elif "toolResult" in content:
                    tr = content["toolResult"]
                    # Flatten tool result content to a string
                    output_parts = []
                    for c in tr.get("content", []):
                        if "text" in c:
                            output_parts.append(c["text"])
                        elif "json" in c:
                            output_parts.append(json.dumps(c["json"]))
                    tool_results.append({
                        "type": "function_call_output",
                        "call_id": tr["toolUseId"],
                        "output": "\n".join(output_parts) if output_parts else "",
                    })

            # Emit text message if present
            if text_parts:
                items.append({
                    "role": role,
                    "content": "\n".join(text_parts),
                })

            # Emit tool calls (always from assistant)
            items.extend(tool_uses)

            # Emit tool results
            items.extend(tool_results)

        return items

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format an OpenAI compatible chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            An OpenAI compatible chat streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to an OpenAI-compatible
                format.
        """
        if not self.use_responses_api:
            request = super().format_request(
                messages,
                tool_specs,
                system_prompt,
                tool_choice,
                system_prompt_content=system_prompt_content,
                **kwargs,
            )
            for msg in request.get("messages", []):
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                reasoning_parts: list[str] = []
                kept: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        reasoning_parts.append(block.get("thinking", ""))
                    else:
                        kept.append(block)
                if reasoning_parts and self.include_reasoning_in_history:
                    msg[self._reasoning_field_name] = "".join(reasoning_parts)
                if len(kept) == 0:
                    kept = [{"type": "text", "text": ""}]
                msg["content"] = kept
            return request
        input_items = self._convert_messages_to_input(messages, system_prompt, chain=bool(self._previous_response_id))
        return {
            "input": input_items,
            "model": self.config["model_id"],
            "stream": True,
            "instructions": system_prompt,
            **({"previous_response_id": self._previous_response_id} if self._previous_response_id else {}),
            "tools": [
                {
                    "type": "function",
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "parameters": tool_spec["inputSchema"]["json"],
                }
                for tool_spec in tool_specs or []
            ],
            **(self._format_request_tool_choice(tool_choice)),
            **cast(dict[str, Any], self.config.get("params", {})),
        }

    @override
    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        if event.get("chunk_type") != "metadata":
            return super().format_chunk(event, **kwargs)

        usage = event["data"]
        usage_dict: dict[str, Any] = {
            "inputTokens": getattr(usage, "prompt_tokens", None),
            "outputTokens": getattr(usage, "completion_tokens", None),
            "totalTokens": getattr(usage, "total_tokens", None),
        }
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
        if cached_tokens is not None:
            usage_dict["cacheReadInputTokens"] = cached_tokens
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
        if reasoning_tokens is None:
            # Together AI puts reasoning_tokens flat on `usage` (in model_extra)
            # instead of nesting it under completion_tokens_details.
            reasoning_tokens = getattr(usage, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            usage_dict["reasoningTokens"] = reasoning_tokens
        return {
            "metadata": {
                "usage": usage_dict,
                "metrics": {"latencyMs": 0},
            },
        }

    def _rlog_begin(self, messages: Messages, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Start a per-call telemetry record and mint the request_id.

        The request_id is injected into the outgoing request body (vLLM carries
        it through the engine into its per-request stats), so client and server
        records of the same call can be joined offline.
        """
        invocation_state = kwargs.get("invocation_state") or {}
        meta = invocation_state.get("request_meta") or {}
        self._call_seq += 1
        rec: dict[str, Any] = {
            "seq": self._call_seq,
            "ts_send": time.time(),
            "n_messages": len(messages),
        }
        self._current_request_id = None
        if self.request_log and meta.get("output_dir"):
            request_id = "{}.{}.{}.{}.c{:04d}".format(
                meta.get("bench", "na"),
                meta.get("arm", "na"),
                meta.get("task_id", "na"),
                self._run_token,
                self._call_seq,
            )
            self._current_request_id = request_id
            rec["request_id"] = request_id
            rec["_path"] = os.path.join(meta["output_dir"], "requests.jsonl")
        return rec

    def _rlog_observe(self, rec: dict[str, Any], chunk: StreamEvent) -> None:
        """Fold a stream chunk into the telemetry record; fix up latencyMs."""
        now = time.time()
        if "ts_first_chunk" not in rec and "messageStart" not in chunk:
            rec["ts_first_chunk"] = now
        metadata = chunk.get("metadata")
        if metadata:
            usage = metadata.get("usage") or {}
            rec["prompt_tokens"] = usage.get("inputTokens")
            rec["completion_tokens"] = usage.get("outputTokens")
            rec["total_tokens"] = usage.get("totalTokens")
            if "cacheReadInputTokens" in usage:
                rec["cached_tokens"] = usage["cacheReadInputTokens"]
            if "reasoningTokens" in usage:
                rec["reasoning_tokens"] = usage["reasoningTokens"]
            # format_chunk hardcodes latencyMs=0; overwrite with the measured
            # wall-clock so accumulated_latency_ms in metrics.json is real.
            metadata.setdefault("metrics", {})["latencyMs"] = int(
                (now - rec["ts_send"]) * 1000
            )
        message_stop = chunk.get("messageStop")
        if message_stop:
            rec["finish_reason"] = message_stop.get("stopReason")

    def _rlog_end(self, rec: dict[str, Any]) -> None:
        self._current_request_id = None
        path = rec.pop("_path", None)
        rec["ts_end"] = time.time()
        rec["latency_ms"] = int((rec["ts_end"] - rec["ts_send"]) * 1000)
        if "ts_first_chunk" in rec:
            rec["ttft_ms"] = int((rec.pop("ts_first_chunk") - rec["ts_send"]) * 1000)
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            LOG.warning("failed to append request log %s: %s", path, e)

    async def _stream_chat_completions(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        rec = self._rlog_begin(messages, kwargs)
        try:
            async for chunk in self._stream_chat_completions_impl(
                messages,
                tool_specs,
                system_prompt,
                tool_choice=tool_choice,
                **kwargs,
            ):
                self._rlog_observe(rec, chunk)
                yield chunk
        except BaseException as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            raise
        finally:
            self._rlog_end(rec)

    async def _stream_chat_completions_impl(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        # Accepts `delta.reasoning` (vLLM/Qwen3) in addition to `delta.reasoning_content`
        LOG.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        LOG.debug("formatted request=<%s>", request)
        if self._current_request_id:
            # Copy: request["extra_body"] may alias the shared params config.
            extra_body = dict(request.get("extra_body") or {})
            extra_body["request_id"] = self._current_request_id
            request["extra_body"] = extra_body

        async with self._get_client() as client:
            extra_headers = {"X-Session-Id": self.session_id} if self.session_id else None
            response = await client.chat.completions.create(**request, extra_headers=extra_headers)
            replica = response.response.headers.get("x-replica")
            if replica:
                LOG.debug("x-replica: %s", replica)

            yield self.format_chunk({"chunk_type": "message_start"})

            tool_calls: dict[int, list[Any]] = {}
            data_type: str | None = None
            finish_reason: str | None = None
            event: Any = None
            self._last_stream_token_ids = []

            async for event in response:
                # LOG.info(f"{event}")
                if not getattr(event, "choices", None):
                    continue
                choice = event.choices[0]
                token_ids = getattr(choice, "token_ids", None)
                if token_ids:
                    self._last_stream_token_ids.extend(token_ids)

                reasoning_delta = getattr(choice.delta, "reasoning_content", None)
                if reasoning_delta:
                    self._reasoning_field_name = "reasoning_content"
                else:
                    reasoning_delta = getattr(choice.delta, "reasoning", None)
                    if reasoning_delta:
                        self._reasoning_field_name = "reasoning"
                if reasoning_delta:
                    chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": data_type,
                            "data": reasoning_delta,
                        }
                    )

                if choice.delta.content:
                    chunks, data_type = self._stream_switch_content("text", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {"chunk_type": "content_delta", "data_type": data_type, "data": choice.delta.content}
                    )

                for tool_call in choice.delta.tool_calls or []:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    if data_type:
                        yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                    break

            for tool_deltas in tool_calls.values():
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})
                for tool_delta in tool_deltas:
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            if self._last_stream_token_ids:
                LOG.info(
                    "stream decoded (n=%d):\n%s",
                    len(self._last_stream_token_ids),
                    "".join(_decode_harmony(self._last_stream_token_ids)),
                )

            if tool_calls and finish_reason != "tool_calls":
                finish_reason = "tool_calls"
            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason or "end_turn"})

            async for event in response:
                _ = event

            if event and hasattr(event, "usage") and event.usage:
                yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})

    async def _process_stream(self, response: Any) -> AsyncGenerator[StreamEvent, None]:
        """Process the streaming response events from the OpenAI Responses API.

        Args:
            response: The async streaming response from OpenAI.

        Yields:
            Formatted stream event chunks.
        """
        tool_calls: dict[str, dict[str, Any]] = {}
        item_id_to_call_id: dict[str, str] = {}  # item.id -> call_id
        current_content_type: str | None = None  # "text" or "reasoning"

        async for event in response:
            event_type = event.type

            # ── Gateway error mid-stream ──
            if event_type == "error":
                code = getattr(event, "code", None)
                msg_text = getattr(event, "message", None) or ""
                LOG.warning("ResponseErrorEvent: code=%s message=%s", code, msg_text[:200])
                self._last_error_code = code
                self._last_error_message = msg_text
                if current_content_type is not None:
                    yield {"contentBlockStop": {}}
                    current_content_type = None
                yield {"messageStop": {"stopReason": "end_turn"}}
                return

            # ── Text content ──
            if event_type == "response.output_text.delta":
                if current_content_type != "text":
                    if current_content_type is not None:
                        yield {"contentBlockStop": {}}
                    yield {"contentBlockStart": {"start": {}}}
                    current_content_type = "text"
                yield {"contentBlockDelta": {"delta": {"text": event.delta}}}

            # ── Reasoning content ──
            elif event_type == "response.reasoning_text.delta":
                if current_content_type != "reasoning":
                    if current_content_type is not None:
                        yield {"contentBlockStop": {}}
                    yield {"contentBlockStart": {"start": {}}}
                    current_content_type = "reasoning"
                yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event.delta}}}}

            # ── Function call start ──
            elif event_type == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    call_id = item.call_id
                    item_id = getattr(item, "id", None)
                    if item_id:
                        item_id_to_call_id[item_id] = call_id
                    tool_calls[call_id] = {
                        "name": item.name,
                        "arguments": "",
                    }

            # ── Function call arguments delta ──
            elif event_type == "response.function_call_arguments.delta":
                call_id = item_id_to_call_id.get(event.item_id, event.item_id)
                if call_id in tool_calls:
                    tool_calls[call_id]["arguments"] += event.delta

            # ── Response completed ──
            elif event_type == "response.completed":
                # Close any open content block
                if current_content_type is not None:
                    yield {"contentBlockStop": {}}
                    current_content_type = None

                # Emit tool call blocks
                for call_id, tc_data in tool_calls.items():
                    yield {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": tc_data["name"],
                                    "toolUseId": call_id,
                                }
                            }
                        }
                    }
                    yield {
                        "contentBlockDelta": {
                            "delta": {
                                "toolUse": {
                                    "input": tc_data["arguments"],
                                }
                            }
                        }
                    }
                    yield {"contentBlockStop": {}}

                # Determine stop reason
                resp = event.response

                def _get(obj: Any, key: str) -> Any:
                    if obj is None:
                        return None
                    if isinstance(obj, dict):
                        return obj.get(key)
                    return getattr(obj, key, None)

                self._previous_response_id = _get(resp, "id") if self.use_previous_id else None
                stop_reason = "end_turn"
                if tool_calls:
                    stop_reason = "tool_use"
                elif _get(resp, "status") == "incomplete":
                    stop_reason = "max_tokens"

                yield {"messageStop": {"stopReason": stop_reason}}

                # Emit usage metadata
                usage = _get(resp, "usage")
                if usage:
                    usage_dict: dict[str, Any] = {
                        "inputTokens": _get(usage, "input_tokens"),
                        "outputTokens": _get(usage, "output_tokens"),
                        "totalTokens": _get(usage, "total_tokens"),
                    }
                    input_details = _get(usage, "input_tokens_details")
                    cached_tokens = _get(input_details, "cached_tokens")
                    if cached_tokens is not None:
                        usage_dict["cacheReadInputTokens"] = cached_tokens
                    output_details = _get(usage, "output_tokens_details")
                    reasoning_tokens = _get(output_details, "reasoning_tokens")
                    if reasoning_tokens is not None:
                        usage_dict["reasoningTokens"] = reasoning_tokens
                    yield {
                        "metadata": {
                            "usage": usage_dict,
                            "metrics": {
                                "latencyMs": 0,
                            },
                        }
                    }

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
        """Stream conversation with the OpenAI model.

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
            ModelThrottledException: If the request is throttled by OpenAI (rate limits).
        """
        # Reset transient error state so downstream hooks only see errors from
        # this call.
        self._last_error_code = None
        self._last_error_message = None
        self._last_stream_token_ids = []
        try:
            if not self.use_responses_api:
                async for chunk in self._stream_chat_completions(
                    messages,
                    tool_specs,
                    system_prompt,
                    tool_choice=tool_choice,
                    **kwargs,
                ):
                    yield chunk
                return

            LOG.debug("formatting request")
            request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
            LOG.debug("formatted request=<%s>", request)

            LOG.debug("invoking model")

            async with self._get_client() as client:
                extra_headers = {"X-Session-Id": self.session_id} if self.session_id else None
                response = await client.responses.create(**request, extra_headers=extra_headers)
                replica = response.response.headers.get("x-replica")
                if replica:
                    LOG.debug("x-replica: %s", replica)

                LOG.debug("got response from model")
                yield self.format_chunk({"chunk_type": "message_start"})

                async for chunk in self._process_stream(response):
                    yield chunk
        except openai.AuthenticationError as e:
            # If using a short-lived bearer token, refresh it so the upstream retry
            # picks up a fresh credential.
            if self.refresh_bedrock_token:
                LOG.warning("OpenAI auth error — refreshing Bedrock token before retry")
                try:
                    self._refresh_bedrock_token()
                except Exception as refresh_err:
                    LOG.error("Failed to refresh Bedrock token: %s", refresh_err)
                raise ModelThrottledException(str(e)) from e
            elif self.refresh_gcloud_token:
                LOG.warning("OpenAI auth error — refreshing gcloud token before retry")
                try:
                    self._refresh_gcloud_token()
                except Exception as refresh_err:
                    LOG.error("Failed to refresh gcloud token: %s", refresh_err)
                raise ModelThrottledException(str(e)) from e
            else:
                raise e
        except openai.BadRequestError as e:
            err_code = getattr(e, "code", None)
            err_msg = str(e)
            # Check if this is a context length exceeded error. Some gateways
            # (e.g. litellm-proxied vLLM) return code="400" with the overflow
            # text only in the message, so also match against known patterns.
            if err_code == "context_length_exceeded" or any(
                m in err_msg for m in _EXPANDED_CONTEXT_OVERFLOW_MESSAGES
            ):
                LOG.warning("OpenAI threw context window overflow error")
                raise ContextWindowOverflowException(err_msg) from e
            # Gateway invalid_prompt
            if err_code == "invalid_prompt" or "unexpected tokens remaining in message header" in err_msg:
                LOG.warning("Gateway invalid_prompt error: %s", err_msg[:200])
                self._last_error_code = "invalid_prompt"
                self._last_error_message = err_msg
                yield self.format_chunk({"chunk_type": "message_start"})
                yield self.format_chunk({"chunk_type": "message_stop", "data": "end_turn"})
                return
            # Re-raise other BadRequestError exceptions
            raise ModelThrottledException(str(e)) from e
        except openai.RateLimitError as e:
            # All rate limit errors should be treated as throttling, not context overflow
            # Rate limits (including TPM) require waiting/retrying, not context reduction
            LOG.warning("OpenAI threw rate limit error")
            raise ModelThrottledException(str(e)) from e
        except openai.APIConnectionError as e:
            LOG.warning("OpenAI connection error: %s", e)
            raise ModelThrottledException(str(e)) from e
        except openai.InternalServerError as e:
            LOG.warning("OpenAI internal server error: %s", e)
            raise ModelThrottledException(str(e)) from e
        except openai.APIError as e:
            LOG.warning("OpenAI api error: %s", e)
            error_message = str(e)
            # openai harmony parse glitches are transient — surface a dedicated
            # exception so a hook can request an immediate retry via the event
            # loop's AfterModelCallEvent.retry flag (no throttle backoff).
            if "HarmonyParseError" in error_message:
                if self._last_stream_token_ids:
                    LOG.warning(
                        "HarmonyParseError after decoding n=%d tokens:\n%s",
                        len(self._last_stream_token_ids),
                        "".join(_decode_harmony(self._last_stream_token_ids)),
                    )
                raise HarmonyParseException(error_message) from e
            # Check for alternative context overflow error messages
            if any(overflow_msg in error_message for overflow_msg in _EXPANDED_CONTEXT_OVERFLOW_MESSAGES):
                LOG.warning("context window overflow error detected")
                raise ContextWindowOverflowException(error_message) from e
            raise ModelThrottledException(str(e)) from e
        except httpx.HTTPError as e:
            # Transport-level errors raised mid-stream (e.g. RemoteProtocolError,
            # ReadTimeout) escape the OpenAI SDK's wrapping.
            LOG.warning("httpx transport error during stream: %s", e)
            raise ModelThrottledException(str(e)) from e

        LOG.debug("finished streaming response from model")
