import logging
from typing import Any, TypedDict
from collections.abc import AsyncGenerator

from google import genai
from google.genai.errors import APIError, ClientError, ServerError
from strands.models.gemini import GeminiModel
from strands.types.content import Messages
from strands.types.exceptions import ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.event_loop import Usage
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import override, Unpack, Required

LOG = logging.getLogger(__name__)


class SRGeminiModel(GeminiModel):
    """Custom Gemini model wrapper that maps cached_content_token_count to cacheReadInputTokens."""
    class GeminiConfig(TypedDict, total=False):
        """Configuration options for Gemini models.
        See parent for attribute details
        """

        model_id: Required[str]
        params: dict[str, Any]
        gemini_tools: list[genai.types.Tool]

    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        client_args: dict[str, Any] | None = None,
        **model_config: Unpack[GeminiConfig],
    ):
        client = genai.Client(**(client_args or {})) 
        super().__init__(client=client, client_args=client_args, **model_config)

    @override
    def _format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        if event["chunk_type"] == "metadata":
            usage_metadata = event["data"]
            usage_data: Usage = {
                "inputTokens": usage_metadata.prompt_token_count,
                "outputTokens": usage_metadata.total_token_count - usage_metadata.prompt_token_count,
                "totalTokens": usage_metadata.total_token_count,
            }

            if cached := getattr(usage_metadata, "cached_content_token_count", None):
                usage_data["cacheReadInputTokens"] = cached

            return {
                "metadata": {
                    "usage": usage_data,
                    "metrics": {
                        "latencyMs": 0,
                    },
                },
            }

        return super()._format_chunk(event)

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Wrap parent stream with exception handling for transient Gemini errors."""
        try:
            async for event in super().stream(messages, tool_specs, system_prompt, tool_choice, **kwargs):
                yield event
        except ServerError as e:
            err_message = str(e)
            LOG.warning(f"Gemini server error. Treating as throttling and retrying. Details: {err_message}")
            raise ModelThrottledException(message=err_message) from e
        except ClientError as e:
            err_message = str(e)
            LOG.warning(f"Gemini client error. Treating as throttling and retrying. Details: {err_message}")
            raise ModelThrottledException(message=err_message) from e
        except APIError as e:
            err_message = str(e)
            LOG.warning(f"Gemini API error. Treating as throttling and retrying. Details: {err_message}")
            raise ModelThrottledException(message=err_message) from e
        except TimeoutError as e:
            err_message = str(e)
            LOG.warning(f"Gemini timeout. Treating as throttling and retrying. Details: {err_message}")
            raise ModelThrottledException(message=err_message) from e
        except ConnectionError as e:
            err_message = str(e)
            LOG.warning(f"Gemini connection error. Treating as throttling and retrying. Details: {err_message}")
            raise ModelThrottledException(message=err_message) from e
