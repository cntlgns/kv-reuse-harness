"""vLLM stat-logger plugin: append one JSONL line per finished request.

Loaded automatically by vLLM through the ``vllm.stat_logger_plugins`` entry
point (see vllm/v1/engine/async_llm.py). Activation is gated on the
``SSA_REQUEST_LEDGER_DIR`` environment variable so installing the package
does not change server behavior by itself.

Each line carries the client-supplied ``request_id`` (sent by the SSA harness
via extra_body, prefixed ``chatcmpl-`` by the OpenAI frontend), which joins
this server-side ledger with the harness's per-run ``requests.jsonl``.
"""

import json
import os
import time
from typing import Any

from vllm.v1.metrics.loggers import StatLoggerBase


class RequestLedgerLogger(StatLoggerBase):
    """Writes <SSA_REQUEST_LEDGER_DIR>/ledger-<engine_index>.jsonl."""

    def __init__(self, vllm_config: Any, engine_index: int = 0):
        self.engine_index = engine_index
        self._f = None
        out_dir = os.environ.get("SSA_REQUEST_LEDGER_DIR")
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"ledger-{engine_index}.jsonl")
        # Line-buffered so records survive an ungraceful server shutdown.
        self._f = open(path, "a", buffering=1)

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        if self._f is None or iteration_stats is None:
            return
        now = round(time.time(), 3)
        for fr in iteration_stats.finished_requests:
            self._f.write(
                json.dumps(
                    {
                        "ts": now,
                        "request_id": fr.request_id,
                        "finish_reason": str(fr.finish_reason),
                        "num_prompt_tokens": fr.num_prompt_tokens,
                        "num_cached_tokens": fr.num_cached_tokens,
                        "num_generation_tokens": fr.num_generation_tokens,
                        "e2e_latency": round(fr.e2e_latency, 4),
                        "queued_time": round(fr.queued_time, 4),
                        "prefill_time": round(fr.prefill_time, 4),
                        "inference_time": round(fr.inference_time, 4),
                        "decode_time": round(fr.decode_time, 4),
                        "mean_time_per_output_token": round(
                            fr.mean_time_per_output_token, 5
                        ),
                        "max_tokens_param": fr.max_tokens_param,
                    }
                )
                + "\n"
            )

    def log_engine_initialized(self) -> None:
        pass
