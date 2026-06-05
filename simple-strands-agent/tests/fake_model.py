"""A deterministic fake Model for SSA tests.

Emits the minimum sequence of ``StreamEvent``s required by strands'
``process_stream`` to produce an assistant message — a ``messageStart``,
per-content-block start/delta/stop events, a ``messageStop`` carrying the
stop reason, and a ``metadata`` event with usage.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any, Literal

from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


StopReason = Literal[
    "content_filtered",
    "end_turn",
    "guardrail_intervened",
    "interrupt",
    "max_tokens",
    "stop_sequence",
    "tool_use",
]


@dataclass
class FakeTurn:
    """One turn of planned model output.

    - ``text``: optional assistant text block
    - ``tool_calls``: list of (tool_name, tool_use_id, input_dict) tuples
    - ``stop_reason``: Bedrock-style stop reason
    - ``usage``: optional override for token counts
    """

    text: str | None = None
    tool_calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    stop_reason: StopReason = "end_turn"
    usage: dict[str, int] | None = None


def _usage(overrides: dict[str, int] | None = None) -> dict[str, int]:
    base = {
        "inputTokens": 10,
        "outputTokens": 5,
        "totalTokens": 15,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 0,
    }
    if overrides:
        base.update(overrides)
    return base


def turn_to_events(turn: FakeTurn) -> list[StreamEvent]:
    """Convert a FakeTurn into a list of Bedrock-style StreamEvents."""
    events: list[StreamEvent] = [{"messageStart": {"role": "assistant"}}]

    block_idx = 0

    if turn.text is not None:
        events.append({"contentBlockStart": {"contentBlockIndex": block_idx, "start": {}}})
        events.append(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": block_idx,
                    "delta": {"text": turn.text},
                }
            }
        )
        events.append({"contentBlockStop": {"contentBlockIndex": block_idx}})
        block_idx += 1

    for name, tool_use_id, input_dict in turn.tool_calls:
        events.append(
            {
                "contentBlockStart": {
                    "contentBlockIndex": block_idx,
                    "start": {"toolUse": {"name": name, "toolUseId": tool_use_id}},
                }
            }
        )
        events.append(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": block_idx,
                    "delta": {"toolUse": {"input": json.dumps(input_dict)}},
                }
            }
        )
        events.append({"contentBlockStop": {"contentBlockIndex": block_idx}})
        block_idx += 1

    events.append(
        {
            "messageStop": {
                "stopReason": turn.stop_reason,
                "additionalModelResponseFields": None,
            }
        }
    )
    events.append(
        {
            "metadata": {
                "usage": _usage(turn.usage),
                "metrics": {"latencyMs": 10, "timeToFirstByteMs": 5},
            }
        }
    )
    return events


class FakeSSAModel(Model):
    """Deterministic Model that replays a fixed list of turns.

    Call ``n`` of ``stream()`` yields the events for ``turns[n]``. If ``stream``
    is called more times than ``turns`` has entries, the final turn is replayed.

    A turn can also be an ``Exception`` to raise from ``stream`` (useful for
    simulating throttling/max-tokens failures at the model layer).
    """

    def __init__(self, turns: list[FakeTurn | Exception]):
        self._turns = list(turns)
        self._idx = 0
        self.call_count = 0
        self.stream_calls: list[Messages] = []

    def _next_turn(self) -> FakeTurn | Exception:
        if self._idx >= len(self._turns):
            turn = self._turns[-1]
        else:
            turn = self._turns[self._idx]
            self._idx += 1
        return turn

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        self.call_count += 1
        self.stream_calls.append(list(messages))

        turn = self._next_turn()
        if isinstance(turn, Exception):
            raise turn

        for event in turn_to_events(turn):
            yield event

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("FakeSSAModel does not implement structured_output")
