import logging
from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry
from strands.types.exceptions import ContextWindowOverflowException


LOG = logging.getLogger(__name__)


class ContextWindowHook(HookProvider):
    """Gate model calls on the prior cycle's totalTokens.

    Raised before the next model request so the previous cycle (assistant
    message + tool execution + usage metrics) finishes committing. The
    StrandsResolverAgent's catch handler then asks the conversation manager
    to reduce context and retries — without dropping the in-flight turn or
    leaving a zero-usage ghost cycle.
    """

    def __init__(self, max_model_len: int, near_limit_threshold: float):
        self.max_model_len = max_model_len
        self.near_limit_threshold = near_limit_threshold
        self.threshold = max_model_len * near_limit_threshold
        # Track the cycle id we last raised on. Reason: After reduce_context trims
        # messages, the stale cycle metric still shows the high token count
        self._last_warned_cycle_id: str | None = None

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeModelCallEvent, self._check_context)

    def _check_context(self, event: BeforeModelCallEvent) -> None:
        invocations = event.agent.event_loop_metrics.agent_invocations
        if not invocations:
            return
        # cycles[-1] is the freshly started current cycle (zero usage at this
        # point — update_usage runs after the model call)
        found_cycle = None
        total = 0
        for cycle in reversed(invocations[-1].cycles):
            cycle_total = (cycle.usage or {}).get("totalTokens", 0) or 0
            if cycle_total:
                found_cycle = cycle
                total = cycle_total
                break
        if found_cycle is None or total < self.threshold:
            return
        if found_cycle.event_loop_cycle_id == self._last_warned_cycle_id:
            # Already raised on this cycle; reduce_context has run. Let the
            # next model call proceed with the trimmed context.
            return
        self._last_warned_cycle_id = found_cycle.event_loop_cycle_id
        msg = (
            f"approaching context window: total_tokens={total} >= "
            f"{int(self.threshold)} (max_model_len={self.max_model_len}, "
            f"near_limit_threshold={self.near_limit_threshold})"
        )
        LOG.warning(msg)
        raise ContextWindowOverflowException(msg)
