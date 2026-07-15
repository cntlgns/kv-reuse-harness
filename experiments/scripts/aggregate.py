#!/usr/bin/env python3
"""Aggregate SSA context-mode experiment results into CSV + per-arm SWE-bench preds.

Usage: python experiments/scripts/aggregate.py [--results DIR]

Outputs (under --results, default experiments/results/ssa):
  summary.csv           one row per run (bench, arm, instance, status, tokens, edits, ...)
  preds_sbv_<ARM>.jsonl SWE-bench prediction files, one per arm, for official scoring
"""
import argparse
import csv
import json
import re
from pathlib import Path

ARMS = ["A", "B1", "B2", "C"]
BENCHES = ["sbv", "tb2"]

TRIM_RE = re.compile(r"Trimming message window")
CTXWIN_RE = re.compile(r"approaching context window")
SHADOW_RE = re.compile(r"tool results truncated|found message with tool results")


def scan_logs(run_dir: Path) -> dict:
    counts = {"trim_events": 0, "ctxwin_events": 0}
    for log in list(run_dir.glob("*.log")) + list(run_dir.glob("run.log")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        counts["trim_events"] += len(TRIM_RE.findall(text))
        counts["ctxwin_events"] += len(CTXWIN_RE.findall(text))
    return counts


def read_metrics(run_dir: Path) -> dict:
    f = run_dir / "metrics.json"
    out = {"input_tokens": "", "output_tokens": "", "total_tokens": "", "llm_calls": ""}
    if not f.exists():
        return out
    try:
        data = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return out
    # metrics.json layout: accumulated usage summary (defensive key lookup)
    def find_key(d, *names):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in names and isinstance(v, (int, float)):
                    return v
                r = find_key(v, *names)
                if r is not None:
                    return r
        elif isinstance(d, list):
            for item in d:
                r = find_key(item, *names)
                if r is not None:
                    return r
        return None

    out["input_tokens"] = find_key(data, "inputTokens", "input_tokens") or ""
    out["output_tokens"] = find_key(data, "outputTokens", "output_tokens") or ""
    out["total_tokens"] = find_key(data, "totalTokens", "total_tokens") or ""
    # per-call lists: length = number of model calls
    per_call = data.get("input_tokens")
    out["llm_calls"] = len(per_call) if isinstance(per_call, list) else ""
    out["wall_sec"] = data.get("execution_time_seconds", "")
    out["cache_hit_rate"] = data.get("cache_hit_rate", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results/ssa")
    args = ap.parse_args()
    res = Path(args.results)

    rows = []
    preds = {arm: [] for arm in ARMS}
    for bench in BENCHES:
        for arm in ARMS:
            base = res / bench / arm
            if not base.is_dir():
                continue
            for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                inst = run_dir.name
                status = (
                    "done" if (run_dir / "DONE").exists()
                    else "failed" if (run_dir / "FAILED").exists()
                    else "incomplete"
                )
                row = {"bench": bench, "arm": arm, "instance": inst, "status": status}
                if bench == "tb2":
                    reward_f = run_dir / "reward.txt"
                    row["reward"] = reward_f.read_text().strip() if reward_f.exists() else ""
                else:
                    patch_f = run_dir / "sra_patch.patch"
                    row["patch_bytes"] = patch_f.stat().st_size if patch_f.exists() else 0
                    pred_f = run_dir / "sra_patch.json"
                    if pred_f.exists():
                        try:
                            raw = json.loads(pred_f.read_text())
                            # sra_patch.json nests the pred dict under the instance id:
                            # {"<inst>": {"instance_id": ..., "model_patch": ...}}
                            pred = raw.get(inst, raw) if isinstance(raw, dict) else None
                            if isinstance(pred, dict) and "model_patch" in pred:
                                pred["model_name_or_path"] = f"ssa-qwen3-coder-30b-{arm}"
                                preds[arm].append(pred)
                        except (OSError, json.JSONDecodeError):
                            pass
                row.update(read_metrics(run_dir))
                row.update(scan_logs(run_dir))
                rows.append(row)

    fields = ["bench", "arm", "instance", "status", "reward", "patch_bytes",
              "input_tokens", "output_tokens", "total_tokens", "llm_calls",
              "wall_sec", "cache_hit_rate", "trim_events", "ctxwin_events"]
    out_csv = res / "summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    for arm, plist in preds.items():
        if plist:
            out = res / f"preds_sbv_{arm}.jsonl"
            with out.open("w") as f:
                for p in plist:
                    f.write(json.dumps(p) + "\n")
            print(f"wrote {out} ({len(plist)} preds)")

    # quick pass-rate table for tb2.
    # Denominator includes FAILED runs (loop-exhaustion/timeout = the policy could
    # not finish the task) so budget-constrained arms aren't inflated by dropping them.
    print("\ntb2 pass rate (reward==1 / attempted):")
    for arm in ARMS:
        sub = [r for r in rows if r["bench"] == "tb2" and r["arm"] == arm
               and r["status"] in ("done", "failed")]
        if sub:
            passed = sum(1 for r in sub if str(r.get("reward", "")).strip() == "1")
            n_fail = sum(1 for r in sub if r["status"] == "failed")
            print(f"  {arm}: {passed}/{len(sub)} (incl. {n_fail} failed-as-0)")


if __name__ == "__main__":
    main()
