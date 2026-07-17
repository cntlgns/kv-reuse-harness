#!/bin/bash
#SBATCH -J vllm-qwen3-coder
#SBATCH -p h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH -o /data_fast/home/sihun/kvcache/kv-reuse-harness/experiments/results/serve/%j.log

# Serve Qwen3-Coder-30B-A3B-Instruct from the research vLLM fork (research/base @ v0.24.0).
# Launch: sbatch experiments/scripts/serve_qwen3_coder_h100.sbatch [port]
# cobra has 8 GPUs, so two 4-GPU instances can coexist on distinct ports.
PORT="${1:-8123}"

export HF_HOME=/data_fast/home/sihun/.cache/huggingface
export HF_HUB_OFFLINE=1          # model fully cached -> no network needed

VLLM_ROOT=/data_fast/home/sihun/kvcache/vllm
VLLM_BIN=$VLLM_ROOT/.venv/bin/vllm
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness

# Per-job telemetry dir: request ledger (stat-logger plugin), /metrics scrapes,
# 1 Hz GPU samples. Everything in here is timestamped for offline joins.
RUN_DIR=$REPO/experiments/results/serve/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
export SSA_REQUEST_LEDGER_DIR=$RUN_DIR

echo "==== vLLM SERVE host=$(hostname) ip=$(hostname -I | awk '{print $1}') port=$PORT $(date -u +%H:%M:%S) ===="
echo "==== vllm=$(git -C $VLLM_ROOT describe --dirty --tags) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) ===="
echo "==== serve_start_epoch=$(date +%s) run_dir=$RUN_DIR ===="

python3 $REPO/experiments/scripts/scrape_metrics.py \
  --url http://localhost:$PORT/metrics \
  --out "$RUN_DIR/metrics_scrape.jsonl" --interval 10 \
  2>> "$RUN_DIR/sidecar.err" &
SCRAPE_PID=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv -l 1 >> "$RUN_DIR/gpu_samples.csv" 2>> "$RUN_DIR/sidecar.err" &
SMI_PID=$!
trap 'kill $SCRAPE_PID $SMI_PID 2>/dev/null' EXIT

$VLLM_BIN serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name qwen3-coder-30b \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prompt-tokens-details
rc=$?
echo "==== serve_end_epoch=$(date +%s) rc=$rc ===="
exit $rc
