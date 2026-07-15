#!/usr/bin/env python3
"""Periodically scrape Prometheus text endpoints into JSONL.

One JSON object per scrape: {"ts": <epoch>, "url": ..., "metrics": {series: value}}.
Series keys keep their label sets verbatim (e.g. 'vllm:prompt_tokens_by_source_total{...,source="local_compute"}').

Usage:
  scrape_metrics.py --out metrics.jsonl [--url http://localhost:8123/metrics]
                    [--interval 10] [--prefix vllm: --prefix lmcache:]

Runs until killed. Stdlib only, so it can run under any python3.
"""

import argparse
import json
import sys
import time
import urllib.request

DEFAULT_PREFIXES = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:num_preemptions",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:e2e_request_latency_seconds",
    "lmcache:",
]


def scrape(url: str, prefixes: list[str]) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode()
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not any(line.startswith(p) for p in prefixes):
            continue
        series, _, value = line.rpartition(" ")
        try:
            metrics[series] = float(value)
        except ValueError:
            continue
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", default=None,
                    help="metrics endpoint(s); default http://localhost:8123/metrics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--prefix", action="append", default=None,
                    help="metric-name prefixes to keep (default: curated vllm:/lmcache: set)")
    args = ap.parse_args()
    urls = args.url or ["http://localhost:8123/metrics"]
    prefixes = args.prefix or DEFAULT_PREFIXES

    with open(args.out, "a", buffering=1) as f:
        while True:
            start = time.time()
            for url in urls:
                try:
                    metrics = scrape(url, prefixes)
                except Exception as e:  # endpoint may be down mid-restart; keep going
                    print(f"[scrape_metrics] {url}: {e}", file=sys.stderr)
                    continue
                f.write(json.dumps(
                    {"ts": round(start, 3), "url": url, "metrics": metrics},
                    separators=(",", ":")) + "\n")
            time.sleep(max(0.0, args.interval - (time.time() - start)))


if __name__ == "__main__":
    sys.exit(main())
