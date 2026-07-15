#!/usr/bin/env python3
"""Analyze one load_driver run: join client/server records, report cost metrics.

Joins, per LLM call, the harness-side requests.jsonl (tasks/*/requests.jsonl,
keyed by request_id) with the vLLM-side ledger (ledger-0.jsonl from the
ssa-request-ledger plugin, request_id prefixed 'chatcmpl-'), restricted to the
steady-state window [run_start + warmup, launch_stopped].

The accounting unit is the TASK (benchmark instance execution), not the
request: context-editing policies that spend extra turns to finish an instance
are automatically penalized in tasks/hour and surfaced in the per-task
turn/token meters, and quality loss is folded in via cost-per-solved-task.

Reports (stdout + <run>/analysis.json, per-task rows in <run>/tasks.csv):
  throughput   tasks/hour/GPU, GPU-seconds/task, and per SOLVED task
  quality      solved rate in window (tb2: reward.txt; sbv: --solved-map)
  per-task     turns (LLM calls), prompt/cached/computed/generated tokens,
               decode-read proxy sum(context_len x generated) per task
  latency      client TTFT and server queue/prefill/decode/TPOT percentiles
  engine       time-averaged running batch, KV usage, preemptions (--scrape)

Usage:
  analyze_load.py --run experiments/results/load/sbv-C-M8-r0 \
      --ledger experiments/results/serve/<jobid>/ledger-0.jsonl \
      [--scrape experiments/results/serve/<jobid>/metrics_scrape.jsonl] \
      [--solved-map solved.json] [--warmup 600] [--gpus 4]

--solved-map: JSON {"<task dir basename>": true/false} from offline scoring
(e.g. official SWE-bench harness on the per-task sra_patch.json predictions).
tb2 runs need no map: tasks/*/reward.txt is read directly.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn tail line from a live writer
    return out


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return round(qs[min(98, max(0, int(q) - 1))], 4)


def dist(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": pct(values, 50),
        "p90": pct(values, 90),
        "p99": pct(values, 99),
    }


def series_value(metrics: dict, prefix: str) -> float | None:
    """Sum all series whose name starts with prefix (collapses label sets)."""
    vals = [v for k, v in metrics.items() if k.startswith(prefix)]
    return sum(vals) if vals else None


def task_solved(task_dir: str, solved_map: dict) -> bool | None:
    name = os.path.basename(task_dir)
    if name in solved_map:
        return bool(solved_map[name])
    reward_path = os.path.join(task_dir, "reward.txt")
    if os.path.exists(reward_path):
        try:
            return float(open(reward_path).read().strip()) >= 1.0
        except ValueError:
            return False
    return None  # unscored (e.g. sbv without --solved-map)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="load_driver --out dir")
    ap.add_argument("--ledger", help="server-side ledger-<engine>.jsonl")
    ap.add_argument("--scrape", help="server-side metrics_scrape.jsonl")
    ap.add_argument("--solved-map", help="JSON {task dir basename: bool} from offline scoring")
    ap.add_argument("--warmup", type=float, default=600.0)
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()

    events = read_jsonl(os.path.join(args.run, "events.jsonl"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    solved_map = json.load(open(args.solved_map)) if args.solved_map else {}
    ledger = {}
    if args.ledger:
        ledger = {r["request_id"]: r for r in read_jsonl(args.ledger)
                  if r.get("request_id")}

    t_start = next((e["ts"] for e in events if e["event"] == "run_start"), None)
    t_stop = next((e["ts"] for e in events if e["event"] == "launch_stopped"), None)
    if t_start is None:
        sys.exit("no run_start event")
    if t_stop is None:  # single-pass run: window ends at last task_end
        t_stop = max((e["ts"] for e in events if e["event"] == "task_end"),
                     default=t_start)
    w0, w1 = t_start + args.warmup, t_stop
    if w1 <= w0:
        sys.exit(f"empty window: warmup {args.warmup}s >= run span {t_stop - t_start:.0f}s")
    window_sec = w1 - w0

    # --- per-task accounting (the unit that penalizes turn inflation) ---
    # A task belongs to the window iff its task_end falls inside it.
    ends_in_window = {e["seq"]: e for e in events
                      if e["event"] == "task_end" and w0 <= e["ts"] <= w1}
    task_rows = []
    all_creqs_w = []
    for task_dir in sorted(glob.glob(os.path.join(args.run, "tasks", "*"))):
        name = os.path.basename(task_dir)
        seq = int(name.split("_", 1)[0])
        end = ends_in_window.get(seq)
        if end is None:
            continue
        creqs = [r for r in read_jsonl(os.path.join(task_dir, "requests.jsonl"))
                 if "request_id" in r] if os.path.exists(
                     os.path.join(task_dir, "requests.jsonl")) else []
        all_creqs_w.extend(creqs)
        joined = [ledger["chatcmpl-" + r["request_id"]] for r in creqs
                  if "chatcmpl-" + r["request_id"] in ledger]
        prompt = sum(s["num_prompt_tokens"] for s in joined)
        cached = sum(s["num_cached_tokens"] for s in joined)
        gen = sum(s["num_generation_tokens"] for s in joined)
        # Agent-phase wall time (MetricsCollector window: after env setup,
        # before verifier/teardown) — splits slot time into agent vs overhead.
        agent_sec = None
        metrics_path = os.path.join(task_dir, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                agent_sec = json.load(open(metrics_path)).get("execution_time_seconds")
            except (json.JSONDecodeError, OSError):
                pass
        row = {
            "task": name,
            "task_id": end.get("task_id"),
            "rc": end.get("rc"),
            "solved": task_solved(task_dir, solved_map),
            "dur_sec": end.get("dur_sec"),
            "agent_sec": agent_sec,
            "overhead_sec": (round(end["dur_sec"] - agent_sec, 1)
                             if agent_sec is not None and end.get("dur_sec") is not None
                             else None),
            "turns": len(creqs),
            "joined": len(joined),
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "computed_prefill_tokens": prompt - cached,
            "generation_tokens": gen,
            # decode-side attention read cost proxy: context length seen by
            # each generated token, summed over the task's requests
            "decode_read_proxy": sum(
                s["num_prompt_tokens"] * s["num_generation_tokens"] for s in joined),
        }
        task_rows.append(row)

    if task_rows:
        with open(os.path.join(args.run, "tasks.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(task_rows[0]))
            writer.writeheader()
            writer.writerows(task_rows)

    # --- throughput & quality-adjusted cost ---
    n_done = len(task_rows)
    n_ok = sum(1 for r in task_rows if r["rc"] == 0)
    scored = [r for r in task_rows if r["solved"] is not None]
    n_solved = sum(1 for r in scored if r["solved"])
    gpu_sec = window_sec * args.gpus
    throughput = {
        "window_sec": round(window_sec, 1),
        "tasks_completed": n_done,
        "tasks_rc0": n_ok,
        "tasks_scored": len(scored),
        "tasks_solved": n_solved,
        "solved_rate": round(n_solved / len(scored), 4) if scored else None,
        "tasks_per_hour_per_gpu": round(n_done / (gpu_sec / 3600), 4) if n_done else 0.0,
        "gpu_sec_per_task": round(gpu_sec / n_done, 1) if n_done else None,
        "gpu_sec_per_solved_task": round(gpu_sec / n_solved, 1) if n_solved else None,
        "task_dur_sec": dist([r["dur_sec"] for r in task_rows]),
        "agent_sec_per_task": dist([r["agent_sec"] for r in task_rows]),
        "overhead_sec_per_task": dist([r["overhead_sec"] for r in task_rows]),
        "turns_per_task": dist([r["turns"] for r in task_rows]),
        "computed_prefill_tokens_per_task": dist(
            [r["computed_prefill_tokens"] for r in task_rows if r["joined"]]),
        "generation_tokens_per_task": dist(
            [r["generation_tokens"] for r in task_rows if r["joined"]]),
        "decode_read_proxy_per_task": dist(
            [r["decode_read_proxy"] for r in task_rows if r["joined"]]),
    }

    # --- request-level latency (window-filtered client records) ---
    creqs_w = [r for r in all_creqs_w if w0 <= r.get("ts_send", 0) <= w1]
    client = {
        "requests_in_window": len(creqs_w),
        "errors_in_window": sum(1 for r in creqs_w if r.get("error")),
        "ttft_ms": dist([r.get("ttft_ms") for r in creqs_w]),
        "latency_ms": dist([r.get("latency_ms") for r in creqs_w]),
        "prompt_tokens": dist([r.get("prompt_tokens") for r in creqs_w]),
        "completion_tokens": dist([r.get("completion_tokens") for r in creqs_w]),
    }

    server = None
    if ledger:
        joined = [ledger["chatcmpl-" + r["request_id"]] for r in creqs_w
                  if "chatcmpl-" + r["request_id"] in ledger]
        prompt = sum(s["num_prompt_tokens"] for s in joined)
        cached = sum(s["num_cached_tokens"] for s in joined)
        tpots = [s["decode_time"] / (s["num_generation_tokens"] - 1)
                 for s in joined if s["num_generation_tokens"] > 1]
        server = {
            "join_rate": round(len(joined) / len(creqs_w), 4) if creqs_w else None,
            "prompt_tokens_sum": prompt,
            "cached_tokens_sum": cached,
            "computed_prefill_tokens_sum": prompt - cached,
            "cached_fraction": round(cached / prompt, 4) if prompt else None,
            "generation_tokens_sum": sum(s["num_generation_tokens"] for s in joined),
            "queued_time_s": dist([s["queued_time"] for s in joined]),
            "prefill_time_s": dist([s["prefill_time"] for s in joined]),
            "decode_time_s": dist([s["decode_time"] for s in joined]),
            "tpot_s": dist(tpots),
        }

    engine = None
    if args.scrape:
        scrapes = [s for s in read_jsonl(args.scrape) if w0 <= s["ts"] <= w1]
        running, kv_usage = [], []
        first, last = None, None
        for s in scrapes:
            m = s["metrics"]
            v = series_value(m, "vllm:num_requests_running")
            if v is not None:
                running.append(v)
            v = series_value(m, "vllm:kv_cache_usage_perc")
            if v is not None:
                kv_usage.append(v)
            last = m
            first = first or m
        def delta(prefix):
            if not first or not last:
                return None
            a, b = series_value(first, prefix), series_value(last, prefix)
            return None if a is None or b is None else b - a
        hits, queries = delta("vllm:prefix_cache_hits"), delta("vllm:prefix_cache_queries")
        engine = {
            "scrapes_in_window": len(scrapes),
            "running_batch": dist(running),
            "kv_cache_usage": dist(kv_usage),
            "kv_cache_usage_max": max(kv_usage) if kv_usage else None,
            "preemptions_delta": delta("vllm:num_preemptions"),
            "prefix_cache_hit_rate": (round(hits / queries, 4)
                                      if hits is not None and queries else None),
        }

    result = {"meta": meta, "warmup_sec": args.warmup,
              "throughput": throughput, "client": client,
              "server": server, "engine": engine}
    out_path = os.path.join(args.run, "analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\n[analyze_load] analysis -> {out_path}; per-task rows -> tasks.csv",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
