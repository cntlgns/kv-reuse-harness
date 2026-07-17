#!/bin/bash
#SBATCH -J score-exp2b
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#SBATCH -t 12:00:00
#SBATCH -o /data_fast/home/sihun/kvcache/kv-reuse-harness/experiments/results/load/logs/%x-%j.log

# Score one or more exp2b preds files sequentially on one node.
# usage: sbatch -p docker -w <node> score_load_preds.sbatch <preds1.jsonl> [preds2.jsonl ...]
set -uo pipefail
SELF="$(readlink -f "$0")"; REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
if ! docker info >/dev/null 2>&1 && getent group docker | grep -qw "$USER"; then
  exec sg docker -c "bash '$SELF' $*"
fi
docker info >/dev/null 2>&1 || { echo "docker FAIL on $(hostname)"; exit 1; }
export PATH="$PATH:/data_fast/home/sihun/miniconda3/bin"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
export HF_HOME="$REPO/experiments/data/hf_home"
OUT="$REPO/experiments/results/load/quality/scoring"
mkdir -p "$OUT"; cd "$OUT"
MAXW="${MAXW:-4}"; TIMEOUT_S="${TIMEOUT_S:-2400}"

for PREDS in "$@"; do
  [ -f "$PREDS" ] || { echo "no preds: $PREDS"; continue; }
  RUN_ID=$(basename "$PREDS" .jsonl | sed 's/^preds_//')
  echo "==== scoring $RUN_ID host=$(hostname) n=$(wc -l < "$PREDS") $(date -u +%H:%M:%S) ===="
  conda run -n swebench_hal python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path "$PREDS" \
    --max_workers "$MAXW" \
    --timeout "$TIMEOUT_S" \
    --cache_level env \
    --run_id "exp2b_${RUN_ID}"
  echo "==== scoring $RUN_ID done rc=$? $(date -u +%H:%M:%S) ===="
done
