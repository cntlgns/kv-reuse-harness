#!/bin/bash
# run_shard.sh <bench:sbv|tb2> <shard_file> — instance-major worker:
# for each instance in the shard, run all 4 context arms sequentially, then
# remove the instance's docker image to bound disk churn (hal-harness pattern).
# Re-execs under `sg docker` so the daemon is reachable inside the sbatch step.
set -uo pipefail
SELF="$(readlink -f "$0")"; SCRIPTS="$(dirname "$SELF")"
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
if ! docker info >/dev/null 2>&1 && getent group docker | grep -qw "$USER"; then
  exec sg docker -c "bash '$SELF' $*"
fi
docker info >/dev/null 2>&1 || { echo "docker unavailable on $(hostname)"; exit 1; }

BENCH="${1:?bench}"; LIST="${2:?shard file}"
ARMS=(A B1 B2 C)
echo "==== shard $BENCH $(basename "$LIST") host=$(hostname) n=$(wc -l < "$LIST") $(date -u +%H:%M:%S) ===="

while IFS= read -r INST; do
  [ -z "$INST" ] && continue
  for ARM in "${ARMS[@]}"; do
    bash "$SCRIPTS/run_one.sh" "$BENCH" "$ARM" "$INST"
  done
  # Disk hygiene: drop this instance's image (harmless no-op if still in use elsewhere)
  if [ "$BENCH" = sbv ]; then
    org="${INST%%__*}"; rest="${INST#*__}"
    docker rmi "swebench/sweb.eval.x86_64.${org}_1776_${rest}" >/dev/null 2>&1 || true
  elif [ "$BENCH" = tb2 ]; then
    img=$(python3 -c "import json,sys; print(json.load(open('$REPO/simple-strands-agent/resources/tb2_docker_public.json')).get('$INST',''))" 2>/dev/null)
    [ -n "$img" ] && docker rmi "$img" >/dev/null 2>&1 || true
  fi
done < "$LIST"
echo "==== shard complete $BENCH $(basename "$LIST") $(date -u +%H:%M:%S) ===="
