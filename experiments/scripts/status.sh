#!/bin/bash
# status.sh — progress table: DONE/FAILED/total per bench x arm, plus SLURM job count.
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
RES="$REPO/experiments/results/ssa"
printf '%-6s %-4s %6s %6s %6s\n' bench arm done failed total
for bench in sbv tb2; do
  case $bench in sbv) total=100 ;; tb2) total=89 ;; esac
  for arm in A B1 B2 C; do
    d=$(find "$RES/$bench/$arm" -name DONE 2>/dev/null | wc -l)
    x=$(find "$RES/$bench/$arm" -name FAILED 2>/dev/null | wc -l)
    printf '%-6s %-4s %6s %6s %6s\n' "$bench" "$arm" "$d" "$x" "$total"
  done
done
echo "---"
squeue -u "$USER" -h -o '%j %T %M %R' | grep -c '^ssa-' | xargs echo "ssa jobs in queue/running:"
squeue -u "$USER" -h -o '%j %T' | grep '^ssa-' | awk '{c[$2]++} END{for (s in c) printf "  %s=%d", s, c[s]; print ""}'
