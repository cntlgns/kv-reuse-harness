#!/bin/bash
# score_sbv.sh <ARM> — score preds_sbv_<ARM>.jsonl with the official swebench harness.
# Modeled on hal-harness swebench_matrix/rescore.sh: score separately from the agent
# phase, NO prune loop, low concurrency (big test suites CPU-starve at high MAXW and
# hit false timeouts), cache_level=env self-bounds disk. One arm per node — two
# cache_level=env scorers on one node can race on shared instance-image removal.
set -uo pipefail
SELF="$(readlink -f "$0")"; REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
if ! docker info >/dev/null 2>&1 && getent group docker | grep -qw "$USER"; then
  exec sg docker -c "bash '$SELF' $*"
fi
docker info >/dev/null 2>&1 || { echo "docker FAIL on $(hostname)"; exit 1; }
ARM="${1:?usage: score_sbv.sh <A|B1|B2|C>}"
export PATH="$PATH:/data_fast/home/sihun/miniconda3/bin"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
# NOTE: ~/.cache/huggingface on /data_fast is corrupted (I/O errors) — use repo-local HF_HOME.
export HF_HOME="$REPO/experiments/data/hf_home"

PREDS="$REPO/experiments/results/ssa/preds_sbv_${ARM}.jsonl"
[ -f "$PREDS" ] || { echo "no preds: $PREDS"; exit 2; }
OUT="$REPO/experiments/results/ssa/scoring"
mkdir -p "$OUT"; cd "$OUT"
MAXW="${MAXW:-4}"; TIMEOUT_S="${TIMEOUT_S:-2400}"

echo "==== scoring arm=$ARM host=$(hostname) n=$(wc -l < "$PREDS") $(date -u +%H:%M:%S) ===="
conda run -n swebench_hal python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$PREDS" \
  --max_workers "$MAXW" \
  --timeout "$TIMEOUT_S" \
  --cache_level env \
  --run_id "ssa_${ARM}"
echo "==== scoring arm=$ARM done $(date -u +%H:%M:%S) ===="
