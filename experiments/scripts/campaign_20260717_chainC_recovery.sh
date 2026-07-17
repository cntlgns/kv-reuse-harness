#!/bin/bash
# chainC recovery — 45740 해제 직후 타 사용자 잡(46024)이 GPU 선점, 신규 서버 45916이
# PENDING에 걸려 chainC(win10 그룹)가 ABORT됨. 45916은 큐에 유지 중: chainA가 끝나며
# 45914를 내리거나 46024가 끝나면 45916이 시작된다. 이 스크립트는 8123이 응답할 때까지
# (최대 12h) 기다렸다가 win10 셀 3개를 tao에서 순차 실행한다.
set -uo pipefail
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
EXP=$REPO/experiments
CELL=$EXP/scripts/load_cell.sbatch
SRV_JOB=45916
PORT=8123
NODE=tao
URL="http://cobra:$PORT/v1"
LOG() { echo "[$(date -u +%m-%d\ %H:%M:%S)] [chainC-rec] $*"; }

wait_job() { local id=$1; while squeue -h -j "$id" 2>/dev/null | grep -q .; do sleep 60; done; }

run_cell() {
  local bench=$1 arm=$2 M=$3 dur=$4 name=$5 manifest=$6
  if [ -f "$EXP/results/load/$name/meta.json" ]; then LOG "SKIP cell $name (exists)"; return 0; fi
  local jid
  jid=$(sbatch --parsable -w "$NODE" "$CELL" "$bench" "$arm" "$M" "$dur" "$name" "$manifest" "$URL") \
    || { LOG "ERROR: sbatch failed for $name"; return 1; }
  LOG "CELL START $name job=$jid node=$NODE arm=$arm M=$M dur=${dur}s"
  wait_job "$jid"
  local n_end=0
  [ -f "$EXP/results/load/$name/events.jsonl" ] && n_end=$(grep -c task_end "$EXP/results/load/$name/events.jsonl" || echo 0)
  LOG "CELL DONE $name job=$jid completions=$n_end"
}

LOG "waiting for server on port $PORT (job $SRV_JOB pending; frees when chainA releases 45914 or 46024 ends)"
t=0
until curl -sf --max-time 5 "http://cobra:$PORT/v1/models" 2>/dev/null | grep -q qwen3; do
  # 45916이 큐에서 사라졌는데 서버가 안 떠 있으면 (취소/실패) 재제출
  if ! squeue -h -j "$SRV_JOB" 2>/dev/null | grep -q .; then
    LOG "server job $SRV_JOB left queue without serving; resubmitting"
    SRV_JOB=$(sbatch --parsable "$EXP/scripts/serve_qwen3_coder_h100.sbatch" "$PORT")
    LOG "SERVER resubmit port=$PORT new=$SRV_JOB"
  fi
  sleep 60; t=$((t+60))
  [ $t -gt 43200 ] && { LOG "ERROR: server on $PORT not up after 12h; giving up"; exit 1; }
done
LOG "SERVER ready port=$PORT job=$SRV_JOB"

run_cell sbv win10 64 7200 sbv-win10-M64-r0 "$EXP/sbv_load_50.txt"
run_cell sbv win10 96 7200 sbv-win10-M96-r0 "$EXP/sbv_sample_100.txt"
run_cell tb2 win10 32 7200 tb2-win10-M32-r0 "$EXP/tb2_all_89.txt"

scancel "$SRV_JOB" 2>/dev/null
LOG "chain complete (server $SRV_JOB released)"
