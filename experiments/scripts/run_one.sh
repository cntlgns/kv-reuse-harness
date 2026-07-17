#!/bin/bash
# run_one.sh <bench:sbv|tb2> <arm:A|B1|B2|C> <instance_id> — one SSA run, resumable.
# Skips if $OUT/DONE exists; marks FAILED (with rc) otherwise so a resubmit retries it.
set -uo pipefail
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
BENCH="${1:?bench}"; ARM="${2:?arm}"; INST="${3:?instance}"
OUT="$REPO/experiments/results/ssa/$BENCH/$ARM/$INST"
if [ -f "$OUT/DONE" ]; then echo "[skip] $BENCH/$ARM/$INST already done"; exit 0; fi
# SKIP_FAILED=1: treat FAILED as terminal (don't retry policy failures, e.g. tb2 B-arm
# 500-iteration exhaustion -- deterministic re-fail, wastes hours)
if [ -n "${SKIP_FAILED:-}" ] && [ -f "$OUT/FAILED" ]; then
  echo "[skip-failed] $BENCH/$ARM/$INST rc=$(cat "$OUT/FAILED")"; exit 0
fi
mkdir -p "$OUT"; rm -f "$OUT/FAILED"

# NOTE(2026-07-15): ~/.cache/huggingface on /data_fast went bad (I/O errors);
# use a repo-local HF_HOME and the materialized sbv dataset copy instead.
export HF_HOME="$REPO/experiments/data/hf_home"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_SBV_DATASET_OFFLINE_LOCATION="$REPO/experiments/data/sbv_dataset"

case "$BENCH" in
  sbv) CFG=sbv_openai_qwen3_coder ;;
  tb2) CFG=tb2_openai_qwen3_coder
       export TB2_REPO_PATH="$REPO/experiments/data/terminal-bench-2"
       export TB2_ECR_MAP="$REPO/simple-strands-agent/resources/tb2_docker_public.json"
       export TB2_INSTRUCTIONS_MAP="$REPO/simple-strands-agent/resources/tb2_instruction_map.json" ;;
  *) echo "unknown bench $BENCH"; exit 2 ;;
esac
case "$ARM" in
  A)  OVR="" ;;
  B1) OVR="agent.context_window.max_model_len=65536" ;;
  B2) OVR="agent.context_window.max_model_len=32768" ;;
  C)  OVR="agent.conversation_manager.win_len=60" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

cd "$REPO"
export SSA_ARM="$ARM"   # labels request_ids in per-request telemetry
echo "[start] $BENCH/$ARM/$INST $(date -u +%H:%M:%S)"
# shellcheck disable=SC2086
timeout -k 60 10800 .venv/bin/python -m ssa.run \
  --config-name="$CFG" \
  dataset.identifier="$INST" \
  hydra.run.dir="$OUT" \
  $OVR > "$OUT/driver.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  touch "$OUT/DONE"
else
  echo "$rc" > "$OUT/FAILED"
fi
echo "[end rc=$rc] $BENCH/$ARM/$INST $(date -u +%H:%M:%S)"
exit 0  # never kill the shard over one instance
