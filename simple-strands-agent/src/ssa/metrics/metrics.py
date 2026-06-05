"""Raw metrics collection for agent runs."""

import json
import logging
import os
import time

from strands.agent import Agent
from strands.telemetry.metrics import EventLoopMetrics

LOG = logging.getLogger(__name__)


def _filter_summary(summary: dict) -> dict:
    """Remove verbose fields from the summary."""
    summary.pop("traces", None)
    for inv in summary.get("agent_invocations", []):
        inv.pop("cycles", None)
    return summary


class MetricsCollector:
    """Context manager that dumps raw per-cycle metrics."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        self._ttft_ms: list[int | None] = []

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def bind(self, agent: Agent) -> None:
        """Attach to the agent's event_loop_metrics to capture per-cycle TTFT values."""
        elm = agent.event_loop_metrics
        original_update = elm.update_metrics

        def _capturing_update(metrics):
            self._ttft_ms.append(metrics.get("timeToFirstByteMs"))
            original_update(metrics)

        elm.update_metrics = _capturing_update

    def dump(self, agent: Agent) -> None:
        """Extract raw per-cycle token and latency arrays and write to JSON."""
        event_loop_metrics: EventLoopMetrics = agent.event_loop_metrics
        input_tokens = []
        output_tokens = []
        reasoning_tokens = []
        cache_read_input_tokens = []
        cache_write_input_tokens = []

        cache_hit_rate_per_invocation = []
        for invocation in event_loop_metrics.agent_invocations:
            inv_cache_read = 0
            inv_cache_write = 0
            for cycle in invocation.cycles:
                usage = cycle.usage
                input_tokens.append(usage["inputTokens"])
                output_tokens.append(usage["outputTokens"])
                reasoning_tokens.append(usage.get("reasoningTokens", 0))
                read_tokens = usage.get("cacheReadInputTokens", 0)
                write_tokens = usage.get("cacheWriteInputTokens", 0)
                cache_read_input_tokens.append(read_tokens)
                cache_write_input_tokens.append(write_tokens)
                inv_cache_read += read_tokens
                inv_cache_write += write_tokens
            inv_total = inv_cache_read + inv_cache_write
            cache_hit_rate_per_invocation.append(
                round(inv_cache_read / inv_total, 4) if inv_total > 0 else 0.0
            )

        total_cache_read = sum(cache_read_input_tokens)
        total_cache_write = sum(cache_write_input_tokens)
        total_cache_tokens = total_cache_read + total_cache_write
        cache_hit_rate = round(total_cache_read / total_cache_tokens, 4) if total_cache_tokens > 0 else 0.0

        data = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_write_input_tokens": cache_write_input_tokens,
            "cache_hit_rate": cache_hit_rate,
            "cache_hit_rate_per_invocation": cache_hit_rate_per_invocation,
            "cycle_durations_sec": event_loop_metrics.cycle_durations,
            "accumulated_usage": dict(event_loop_metrics.accumulated_usage),
            "ttft_ms": self._ttft_ms,
            "accumulated_latency_ms": event_loop_metrics.accumulated_metrics["latencyMs"],
            "execution_time_seconds": round(time.monotonic() - self._start, 3),
            "summary": _filter_summary(event_loop_metrics.get_summary()),
        }

        metrics_file = os.path.join(self._output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        LOG.info(f"Metrics written to {metrics_file}")

    def __exit__(self, *_):
        return False
