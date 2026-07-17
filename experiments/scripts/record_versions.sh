#!/usr/bin/env bash
# Record exact code versions into an experiment result directory.
#
# Usage: record_versions.sh <output_dir>
# Env:   VLLM_ROOT — vllm 클론 경로 (기본: 하네스 옆의 ../vllm)
set -euo pipefail

OUT="${1:?usage: record_versions.sh <output_dir>}"
HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-$(dirname "$HARNESS_ROOT")/vllm}"

commit()   { git -C "$1" rev-parse HEAD 2>/dev/null || echo "unknown"; }
describe() { git -C "$1" describe --always --dirty --tags 2>/dev/null || echo "unknown"; }
dirty()    { if [ -n "$(git -C "$1" status --porcelain 2>/dev/null)" ]; then echo true; else echo false; fi; }

mkdir -p "$OUT"
cat > "$OUT/versions.json" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "harness": {
    "commit": "$(commit "$HARNESS_ROOT")",
    "describe": "$(describe "$HARNESS_ROOT")",
    "dirty": $(dirty "$HARNESS_ROOT")
  },
  "vllm": {
    "commit": "$(commit "$VLLM_ROOT")",
    "describe": "$(describe "$VLLM_ROOT")",
    "dirty": $(dirty "$VLLM_ROOT")
  }
}
EOF

echo "wrote $OUT/versions.json"
if [ "$(dirty "$HARNESS_ROOT")" = true ] || [ "$(dirty "$VLLM_ROOT")" = true ]; then
  echo "WARNING: uncommitted changes present — this run is not reproducible" >&2
fi
