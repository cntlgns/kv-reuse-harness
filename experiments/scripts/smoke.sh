#!/bin/bash
# smoke.sh — pipeline sanity check on a docker node: 1 sbv + 1 tb2 instance, arm A.
set -uo pipefail
SELF="$(readlink -f "$0")"; SCRIPTS="$(dirname "$SELF")"
if ! docker info >/dev/null 2>&1 && getent group docker | grep -qw "$USER"; then
  exec sg docker -c "bash '$SELF' $*"
fi
docker info >/dev/null 2>&1 || { echo "docker unavailable on $(hostname)"; exit 1; }
echo "==== smoke host=$(hostname) $(date -u +%H:%M:%S) ===="
bash "$SCRIPTS/run_one.sh" sbv A astropy__astropy-13398
bash "$SCRIPTS/run_one.sh" tb2 A adaptive-rejection-sampler
echo "==== smoke done $(date -u +%H:%M:%S) ===="
