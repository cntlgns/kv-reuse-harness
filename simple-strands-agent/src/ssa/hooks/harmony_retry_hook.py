import logging

from strands.hooks import AfterModelCallEvent, HookProvider, HookRegistry

from ssa.models.openai import HarmonyParseException


LOG = logging.getLogger(__name__)


class HarmonyRetryHook(HookProvider):
    """Trigger an immediate model retry when openai models emits a HarmonyParseError.

    The error is server-side and flagged retryable. This hook sets
    `AfterModelCallEvent.retry = True` directly so the event loop's outer
    retry loop fires another model call without delay.
    """

    def __init__(self, max_retries: int = 500) -> None:
        self.max_retries = max_retries
        self._attempts: int = 0

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(AfterModelCallEvent, self._on_after_model_call)

    def _on_after_model_call(self, event: AfterModelCallEvent):
        if event.exception is None:
            self._attempts = 0
            return

        if not isinstance(event.exception, HarmonyParseException):
            return

        self._attempts += 1
        if self._attempts > self.max_retries:
            LOG.error(
                "HarmonyParseError: exhausted %d retries, giving up",
                self.max_retries,
            )
            return

        LOG.warning(
            "HarmonyParseError (attempt %d/%d), requesting model retry",
            self._attempts,
            self.max_retries,
        )
        event.retry = True
