#!/usr/bin/env python3
"""Stationarity diagnostics for load_driver cells.

Per cell:
  - cohort: when did the first M (synchronized) tasks finish vs w0=run_start+600
  - bins:   completion rate in 6 equal bins of the window, normalized to mean
            (wave pattern / trend detector)
  - halves: first-half vs second-half tasks/hour in the window
  - little: 3600*M/mean(dur_sec of window tasks) vs measured tasks/hour
  - rc:     nonzero-exit tasks counted in the window (validity check)
"""
import json
import glob
import os
import statistics
import sys

WARMUP = float(os.environ.get("WARMUP", 600))
BINS = 6

def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out

def analyze(run_dir):
    ev = read_jsonl(os.path.join(run_dir, "events.jsonl"))
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    M = meta["concurrency"]
    t0 = next((e["ts"] for e in ev if e["event"] == "run_start"), None)
    t_stop = next((e["ts"] for e in ev if e["event"] == "launch_stopped"), None)
    if t0 is None or t_stop is None:
        return None
    w0, w1 = t0 + WARMUP, t_stop
    win = w1 - w0
    ends = [e for e in ev if e["event"] == "task_end"]
    ends_w = [e for e in ends if w0 <= e["ts"] <= w1]
    n = len(ends_w)

    # --- initial cohort: seq 1..M ---
    cohort = [e for e in ends if e["seq"] <= M]
    cohort_done = sorted(e["ts"] - t0 for e in cohort)
    n_before_w0 = sum(1 for t in cohort_done if t <= WARMUP)
    cohort_p90 = (cohort_done[int(0.9 * (len(cohort_done) - 1))]
                  if cohort_done else None)
    cohort_max = cohort_done[-1] if cohort_done else None

    # --- binned completion rate (normalized) ---
    counts = [0] * BINS
    for e in ends_w:
        b = min(BINS - 1, int((e["ts"] - w0) / win * BINS))
        counts[b] += 1
    mean_c = n / BINS if n else 0
    norm = [round(c / mean_c, 2) if mean_c else 0 for c in counts]

    # --- halves ---
    h1 = sum(1 for e in ends_w if e["ts"] <= w0 + win / 2)
    h2 = n - h1

    # --- Little's law ---
    durs = [e["dur_sec"] for e in ends_w if e.get("dur_sec")]
    mean_dur = statistics.fmean(durs) if durs else None
    lam_meas = n / (win / 3600)  # tasks/hour, all GPUs
    lam_little = 3600 * M / mean_dur if mean_dur else None

    rc_bad = sum(1 for e in ends_w if e.get("rc") != 0)
    return {
        "cell": os.path.basename(run_dir), "M": M,
        "window_min": round(win / 60, 1), "n": n, "rc_bad": rc_bad,
        "cohort_done_at_w0": f"{n_before_w0}/{M}",
        "cohort_p90_min": round(cohort_p90 / 60, 1) if cohort_p90 else None,
        "cohort_max_min": round(cohort_max / 60, 1) if cohort_max else None,
        "bins_norm": norm,
        "halves": f"{h1}/{h2}",
        "half_ratio": round(h2 / h1, 2) if h1 else None,
        "tph_measured": round(lam_meas, 1),
        "tph_little": round(lam_little, 1) if lam_little else None,
        "little_ratio": round(lam_meas / lam_little, 2) if lam_little else None,
        "mean_dur_min": round(mean_dur / 60, 1) if mean_dur else None,
    }

def main():
    rows = []
    for d in sorted(glob.glob(sys.argv[1])):
        if not os.path.exists(os.path.join(d, "events.jsonl")):
            continue
        r = analyze(d)
        if r:
            rows.append(r)
    hdr = ["cell", "M", "window_min", "n", "rc_bad", "cohort_done_at_w0",
           "cohort_p90_min", "cohort_max_min", "mean_dur_min",
           "halves", "half_ratio", "tph_measured", "tph_little",
           "little_ratio", "bins_norm"]
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r[k]) for k in hdr))

if __name__ == "__main__":
    main()
