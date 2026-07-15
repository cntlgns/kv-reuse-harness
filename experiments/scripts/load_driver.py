#!/usr/bin/env python3
"""Closed-loop load generator for serving-cost experiments.

Keeps exactly M agent tasks in flight against the vLLM server: when one task
finishes, the next manifest entry starts immediately. One invocation = one
measurement cell (one bench x arm x M). Unlike run_one.sh there are no
DONE-resume semantics: every execution gets a fresh task dir, and the manifest
is recycled (reshuffled) when --duration outlasts it, so concurrency never
drains before the measurement window ends.

Output layout (--out DIR):
  meta.json         run parameters, start/end epochs
  versions.json     harness+vllm SHAs (via record_versions.sh)
  events.jsonl      task_start/task_end/sys records with wall-clock timestamps
  tasks/NNNN_<id>/  one hydra run dir per task execution (metrics.json,
                    requests.jsonl, trajectory.json, driver.log, ...)

Example:
  python3 experiments/scripts/load_driver.py \
      --bench sbv --arm C -M 8 --duration 7200 \
      --manifest experiments/sbv_sample_100.txt \
      --out experiments/results/load/sbv-C-M8-r0 --prepull
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time

REPO = "/data_fast/home/sihun/kvcache/kv-reuse-harness"
PYTHON = os.path.join(REPO, ".venv/bin/python")

BENCH_CFG = {
    "sbv": "sbv_openai_qwen3_coder",
    "tb2": "tb2_openai_qwen3_coder",
}
ARM_OVERRIDES = {
    "A": [],
    "B1": ["agent.context_window.max_model_len=65536"],
    "B2": ["agent.context_window.max_model_len=32768"],
    "C": ["agent.conversation_manager.win_len=60"],
}


def task_env(bench: str, arm: str) -> dict:
    env = dict(os.environ)
    env["SSA_ARM"] = arm
    env["HF_HOME"] = f"{REPO}/experiments/data/hf_home"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_SBV_DATASET_OFFLINE_LOCATION"] = f"{REPO}/experiments/data/sbv_dataset"
    if bench == "tb2":
        env["TB2_REPO_PATH"] = f"{REPO}/experiments/data/terminal-bench-2"
        env["TB2_ECR_MAP"] = f"{REPO}/simple-strands-agent/resources/tb2_docker_public.json"
        env["TB2_INSTRUCTIONS_MAP"] = f"{REPO}/simple-strands-agent/resources/tb2_instruction_map.json"
    return env


def docker_image(bench: str, instance_id: str) -> str:
    if bench == "sbv":
        org, rest = instance_id.split("__", 1)
        return f"swebench/sweb.eval.x86_64.{org}_1776_{rest}"
    with open(f"{REPO}/simple-strands-agent/resources/tb2_docker_public.json") as f:
        return json.load(f)[instance_id]


class EventLog:
    def __init__(self, path: str):
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        with self._lock:
            self._f.write(json.dumps(rec) + "\n")


class TaskQueue:
    """Round-robin over the manifest, reshuffled each cycle."""

    def __init__(self, instances: list[str], seed: int, cycle: bool):
        self._instances = list(instances)
        self._cycle = cycle
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._order = list(instances)
        self._rng.shuffle(self._order)
        self._idx = 0
        self._seq = 0

    def next(self) -> tuple[int, str] | None:
        with self._lock:
            if self._idx >= len(self._order):
                if not self._cycle:
                    return None
                self._rng.shuffle(self._order)
                self._idx = 0
            inst = self._order[self._idx]
            self._idx += 1
            self._seq += 1
            return self._seq, inst


def run_task(seq: int, inst: str, slot: int, args, env, events: EventLog,
             active: dict, active_lock: threading.Lock) -> None:
    out = os.path.join(args.out, "tasks", f"{seq:04d}_{inst}")
    os.makedirs(out, exist_ok=True)
    cmd = [
        PYTHON, "-m", "ssa.run",
        f"--config-name={BENCH_CFG[args.bench]}",
        f"dataset.identifier={inst}",
        f"hydra.run.dir={out}",
        "++agent.invoker_params.request_log=true",
        *ARM_OVERRIDES[args.arm],
    ]
    events.emit("task_start", seq=seq, task_id=inst, slot=slot)
    start = time.monotonic()
    with open(os.path.join(out, "driver.log"), "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=REPO, start_new_session=True)
        with active_lock:
            active[seq] = proc
        try:
            rc = proc.wait(timeout=args.task_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
            rc = rc if rc != 0 else 124
        finally:
            with active_lock:
                active.pop(seq, None)
    events.emit("task_end", seq=seq, task_id=inst, slot=slot, rc=rc,
                dur_sec=round(time.monotonic() - start, 1))


def prepull(bench: str, instances: list[str], events: EventLog) -> None:
    images = []
    for inst in dict.fromkeys(instances):
        try:
            images.append(docker_image(bench, inst))
        except Exception as e:
            print(f"[prepull] cannot resolve image for {inst}: {e}", file=sys.stderr)
    for i, img in enumerate(dict.fromkeys(images)):
        print(f"[prepull] ({i + 1}/{len(images)}) {img}")
        rc = subprocess.run(["docker", "pull", "-q", img],
                            stdout=subprocess.DEVNULL).returncode
        if rc != 0:
            print(f"[prepull] FAILED: {img}", file=sys.stderr)
    events.emit("prepull_done", n_images=len(images))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=sorted(BENCH_CFG), required=True)
    ap.add_argument("--arm", choices=sorted(ARM_OVERRIDES), required=True)
    ap.add_argument("-M", "--concurrency", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop launching new tasks after this many seconds; "
                         "0 = single pass through the manifest (no recycling)")
    ap.add_argument("--task-timeout", type=float, default=10800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prepull", action="store_true",
                    help="docker-pull all manifest images before starting the clock")
    ap.add_argument("--stop-mode", choices=["drain", "kill"], default="drain",
                    help="at duration end: 'drain' lets in-flight tasks finish; "
                         "'kill' SIGTERMs them (post-window completions are "
                         "excluded from analysis anyway)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        instances = [ln.strip() for ln in f if ln.strip()]
    os.makedirs(os.path.join(args.out, "tasks"), exist_ok=True)
    events = EventLog(os.path.join(args.out, "events.jsonl"))
    subprocess.run([f"{REPO}/experiments/scripts/record_versions.sh", args.out],
                   check=False)

    meta = {
        "bench": args.bench, "arm": args.arm, "concurrency": args.concurrency,
        "manifest": args.manifest, "n_instances": len(instances),
        "duration": args.duration, "task_timeout": args.task_timeout,
        "seed": args.seed, "host": os.uname().nodename,
        "start_epoch": round(time.time(), 3),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if args.prepull:
        prepull(args.bench, instances, events)

    queue = TaskQueue(instances, args.seed, cycle=args.duration > 0)
    stop_launch = threading.Event()
    active: dict[int, subprocess.Popen] = {}
    active_lock = threading.Lock()
    env = task_env(args.bench, args.arm)

    def worker(slot: int) -> None:
        while not stop_launch.is_set():
            item = queue.next()
            if item is None:
                return
            run_task(item[0], item[1], slot, args, env, events, active, active_lock)

    def monitor() -> None:
        while not stop_launch.is_set():
            with active_lock:
                running = len(active)
            events.emit("sys", running=running, loadavg=os.getloadavg())
            stop_launch.wait(30)

    def on_signal(signum, _frame):
        print(f"[load_driver] signal {signum}: stop launching, draining...",
              file=sys.stderr)
        stop_launch.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    events.emit("run_start", **meta)
    threads = [threading.Thread(target=worker, args=(s,), daemon=True)
               for s in range(args.concurrency)]
    for t in threads:
        t.start()
    threading.Thread(target=monitor, daemon=True).start()

    if args.duration > 0:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline and not stop_launch.is_set():
            time.sleep(5)
        stop_launch.set()
        events.emit("launch_stopped")
        if args.stop_mode == "kill":
            with active_lock:
                procs = list(active.values())
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            events.emit("kill_sent", n=len(procs))
            for t in threads:
                t.join(timeout=180)
            with active_lock:  # escalate on stragglers
                procs = list(active.values())
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    for t in threads:  # drain mode: in-flight tasks run to completion
        t.join()

    events.emit("run_end")
    meta["end_epoch"] = round(time.time(), 3)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[load_driver] done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
