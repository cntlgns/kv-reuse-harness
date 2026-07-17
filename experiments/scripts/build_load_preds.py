#!/usr/bin/env python3
"""Build SWE-bench preds jsonl from load-cell task dirs, split by completion round.

Load cells complete each manifest instance 1..k times (closed-loop recycling).
Round r = the r-th completion of each instance (by seq order), so each round
is a valid one-prediction-per-instance file for the official harness.

usage: build_load_preds.py --arm win30 --cells DIR [DIR ...] --out DIR [--max-rounds 2]
"""
import argparse, json, os, re, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--cells", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-rounds", type=int, default=2)
a = ap.parse_args()

by_inst = defaultdict(list)  # instance -> [(cell_idx, seq, patch_path)]
for ci, cell in enumerate(a.cells):
    tdir = os.path.join(cell, "tasks")
    for name in sorted(os.listdir(tdir)):
        m = re.match(r"(\d+)_(.+)$", name)
        if not m:
            continue
        seq, inst = int(m.group(1)), m.group(2)
        p = os.path.join(tdir, name, "sra_patch.patch")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            by_inst[inst].append((ci, seq, p))

for inst in by_inst:
    by_inst[inst].sort()

os.makedirs(a.out, exist_ok=True)
for r in range(a.max_rounds):
    rows = []
    for inst, lst in sorted(by_inst.items()):
        if r < len(lst):
            patch = open(lst[r][2]).read()
            rows.append({"instance_id": inst,
                         "model_name_or_path": f"exp2b-{a.arm}",
                         "model_patch": patch})
    out = os.path.join(a.out, f"preds_{a.arm}_round{r+1}.jsonl")
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"round{r+1}: {len(rows)} preds -> {out}")
print(f"instances with >=1 completion: {len(by_inst)}")
