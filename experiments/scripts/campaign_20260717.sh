#!/bin/bash
# campaign_20260717.sh — Exp-2b: M96 long-window reruns (A/C) + new arms win30/win10
# + first tb2 cost cells (quality harvested from completions).
#
# Matrix (10 cells, two parallel chains, one client node + one 4-GPU server each):
#   chain A (client david, server port 8124, starts on job 45748 = A-warm):
#     sbv-A-M96-r1      M=96 dur=7200 manifest=sbv_sample_100
#     tb2-A-M32-r0      M=32 dur=7200 manifest=tb2_all_89
#     [server restart]  -> win30 group
#     sbv-win30-M64-r0  M=64 dur=7200 manifest=sbv_load_50
#     sbv-win30-M96-r0  M=96 dur=7200 manifest=sbv_sample_100
#     tb2-win30-M32-r0  M=32 dur=7200 manifest=tb2_all_89
#   chain C (client tao, server port 8123, starts on job 45740 = C-warm):
#     same with C / win10.
#
# Policy: server restart between ARM groups (cache reset); same-arm cells run
# back-to-back on one server (Exp-2 precedent). Quality comes free from task
# dirs: sbv sra_patch.patch (offline scoring), tb2 reward.txt.
set -uo pipefail
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
EXP=$REPO/experiments
SERVE=$EXP/scripts/serve_qwen3_coder_h100.sbatch
CELL=$EXP/scripts/load_cell.sbatch
LOG() { echo "[$(date -u +%m-%d\ %H:%M:%S)] $*"; }

wait_job() {  # <jobid>  block until slurm job leaves the queue
  local id=$1
  while squeue -h -j "$id" 2>/dev/null | grep -q .; do sleep 60; done
}

wait_server() {  # <port>  block until vLLM answers
  local port=$1 t=0
  until curl -sf --max-time 5 "http://cobra:$port/v1/models" 2>/dev/null | grep -q qwen3; do
    sleep 30; t=$((t+30))
    [ $t -gt 3600 ] && { LOG "ERROR: server on $port not up after 60min"; return 1; }
  done
  LOG "server on port $port is ready"
}

disk_guard() {  # <node>  report free space; prune dangling if tight
  local node=$1
  timeout 300 srun -p docker -w "$node" --time=4 -J diskguard bash -c '
    free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    echo "disk[$(hostname)] free=${free}G"
    if [ "$free" -lt 100 ]; then
      sg docker -c "docker image prune -f" >/dev/null 2>&1
      echo "disk[$(hostname)] pruned dangling images"
    fi' 2>/dev/null || LOG "WARN: disk_guard on $node failed (node busy?)"
}

run_cell() {  # <node> <bench> <arm> <M> <dur> <name> <manifest> <base_url>
  local node=$1 bench=$2 arm=$3 M=$4 dur=$5 name=$6 manifest=$7 base_url=$8
  if [ -f "$EXP/results/load/$name/analysis.json" ] || [ -f "$EXP/results/load/$name/meta.json" ]; then
    LOG "SKIP cell $name (already exists)"; return 0
  fi
  local jid
  jid=$(sbatch --parsable -w "$node" "$CELL" "$bench" "$arm" "$M" "$dur" "$name" "$manifest" "$base_url") \
    || { LOG "ERROR: sbatch failed for $name"; return 1; }
  LOG "CELL START $name job=$jid node=$node arm=$arm M=$M dur=${dur}s"
  wait_job "$jid"
  if [ -f "$EXP/results/load/$name/events.jsonl" ]; then
    local n_end
    n_end=$(grep -c task_end "$EXP/results/load/$name/events.jsonl" 2>/dev/null || echo 0)
    LOG "CELL DONE $name job=$jid completions=$n_end"
  else
    LOG "ERROR: CELL $name produced no events.jsonl (job $jid)"
  fi
}

restart_server() {  # <old_jid> <port>  returns new jid on stdout
  local old=$1 port=$2
  scancel "$old" 2>/dev/null; sleep 20
  local jid
  jid=$(sbatch --parsable "$SERVE" "$port")
  LOG "SERVER restart port=$port old=$old new=$jid" >&2
  wait_server "$port" >&2 || return 1
  echo "$jid"
}

chain() {  # <label> <node> <port> <server_jid> <arm1> <arm2>
  local label=$1 node=$2 port=$3 srv=$4 arm1=$5 arm2=$6
  local url="http://cobra:$port/v1"
  LOG "[$label] begin: node=$node port=$port server=$srv arms=$arm1,$arm2"

  # ---- group 1: existing arm, warm server (same-arm continuation) ----
  run_cell "$node" sbv "$arm1" 96 7200 "sbv-$arm1-M96-r1" "$EXP/sbv_sample_100.txt" "$url"
  disk_guard "$node"
  run_cell "$node" tb2 "$arm1" 32 7200 "tb2-$arm1-M32-r0" "$EXP/tb2_all_89.txt" "$url"

  # ---- group 2: new win arm, fresh server ----
  srv=$(restart_server "$srv" "$port") || { LOG "[$label] ABORT: server restart failed"; return 1; }
  disk_guard "$node"
  run_cell "$node" sbv "$arm2" 64 7200 "sbv-$arm2-M64-r0" "$EXP/sbv_load_50.txt" "$url"
  run_cell "$node" sbv "$arm2" 96 7200 "sbv-$arm2-M96-r0" "$EXP/sbv_sample_100.txt" "$url"
  disk_guard "$node"
  run_cell "$node" tb2 "$arm2" 32 7200 "tb2-$arm2-M32-r0" "$EXP/tb2_all_89.txt" "$url"

  scancel "$srv" 2>/dev/null
  LOG "[$label] chain complete (server $srv released)"
}

mkdir -p "$EXP/results/load/logs"
LOG "campaign start: pid=$$"
chain A david 8124 45748 A win30 > >(sed -u 's/^/[chainA] /') 2>&1 &
PA=$!
chain C tao   8123 45740 C win10 > >(sed -u 's/^/[chainC] /') 2>&1 &
PC=$!
wait $PA; RA=$?
wait $PC; RC=$?
LOG "campaign finished: chainA rc=$RA chainC rc=$RC"
exit $(( RA || RC ))
