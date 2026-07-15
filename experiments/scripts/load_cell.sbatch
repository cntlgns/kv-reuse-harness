#!/bin/bash
#SBATCH -J ssa-load
#SBATCH -p docker
#SBATCH --cpus-per-task=44
#SBATCH --mem=200G
#SBATCH -t 8:00:00
#SBATCH -o /data_fast/home/sihun/kvcache/kv-reuse-harness/experiments/results/load/logs/%x-%j.log

# One cost-measurement cell: closed-loop load at fixed (bench, arm, M).
# usage: sbatch -w <node> load_cell.sbatch <bench> <arm> <M> <duration_s> <name> [manifest]
# The whole node should be dedicated: no scoring/other jobs during the window.
set -uo pipefail
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
SELF="$REPO/experiments/scripts/load_cell.sbatch"
if ! docker info >/dev/null 2>&1 && getent group docker | grep -qw "$USER"; then
  exec sg docker -c "bash '$SELF' $*"
fi
docker info >/dev/null 2>&1 || { echo "docker FAIL on $(hostname)"; exit 1; }

BENCH="${1:?bench}"; ARM="${2:?arm}"; M="${3:?M}"; DUR="${4:?duration_s}"; NAME="${5:?name}"
case "$BENCH" in
  sbv) DEFAULT_MANIFEST=$REPO/experiments/sbv_sample_100.txt ;;
  tb2) DEFAULT_MANIFEST=$REPO/experiments/tb2_all_89.txt ;;
  *) echo "unknown bench $BENCH"; exit 2 ;;
esac
MANIFEST="${6:-$DEFAULT_MANIFEST}"

echo "==== load cell host=$(hostname) bench=$BENCH arm=$ARM M=$M dur=${DUR}s name=$NAME ===="
df -h /var/lib/docker / 2>/dev/null | tail -2
exec python3 "$REPO/experiments/scripts/load_driver.py" \
  --bench "$BENCH" --arm "$ARM" -M "$M" --duration "$DUR" \
  --manifest "$MANIFEST" \
  --out "$REPO/experiments/results/load/$NAME" --prepull
