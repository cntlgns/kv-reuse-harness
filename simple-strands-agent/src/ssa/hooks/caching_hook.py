import logging
from typing import List, Any, Literal
from strands.hooks import BeforeModelCallEvent
from strands.hooks import HookProvider, HookRegistry
from strands.types.content import Message


LOG = logging.getLogger(__name__)


class PromptCacheHook(HookProvider):
    def __init__(self, cache_period: int = 5, **kwargs):
        self.cache_period: int = cache_period
        self.cache_marker_ctr: int = 0
        self.cache_ttl: Literal["1h", "5m"] = kwargs.get("ttl", "5m")
        super().__init__()

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeModelCallEvent, self._add_prompt_cache_marker)

    def _add_prompt_cache_marker(self, event: BeforeModelCallEvent) -> None:
        self.cache_marker_ctr += 1
        messages: List[Message] = event.agent.messages
        if self.cache_marker_ctr % self.cache_period == 0:
            self._inject_cache_point(messages)

    def _inject_cache_point(self, messages: list[dict[str, Any]]) -> None:
        """
        Specific to bedrock invoker: 
        Inject a cache point at the end of the last assistant message.
        """
        if not messages:
            return

        last_assistant_idx: int | None = None
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", [])
            for block_idx, block in reversed(list(enumerate(content))):
                if "cachePoint" in block:
                    del content[block_idx]
            if msg.get("role") == "assistant":
                last_assistant_idx = msg_idx

        if last_assistant_idx is not None and messages[last_assistant_idx].get("content"):
            if self.cache_ttl == "1h":
                messages[last_assistant_idx]["content"].append({"cachePoint": {"type": "default", "ttl": self.cache_ttl}})
            else:
                messages[last_assistant_idx]["content"].append({"cachePoint": {"type": "default"}}) 
            LOG.info(f"msg_idx={last_assistant_idx} | added cache point to last assistant message", )
