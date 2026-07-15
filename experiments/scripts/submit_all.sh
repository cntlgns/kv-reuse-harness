#!/bin/bash
# submit_all.sh — shard sbv(100)+tb2(89) instance lists and sbatch them PACKED
# (cpu/mem slice, no --exclusive) across the docker-capable nodes david/tao/ruby,
# following hal-harness swebench_matrix/submit_run.sh. Concurrency = #shards.
#   SHARDS_PER_BENCH=12 (default) -> 24 jobs, 8 per node, 24 concurrent SSA runs.
# Resumable: rerunning submit_all.sh only re-runs instances without a DONE marker.
set -uo pipefail
REPO=/data_fast/home/sihun/kvcache/kv-reuse-harness
EXP="$REPO/experiments"
SHARDS_PER_BENCH="${SHARDS_PER_BENCH:-12}"
NODES=(david tao ruby)
BENCHES=("${@:-sbv tb2}")   # optionally pass a subset: submit_all.sh sbv
[ $# -eq 0 ] && BENCHES=(sbv tb2)
mkdir -p "$EXP/results/ssa/logs" "$EXP/work"

bash "$EXP/scripts/record_versions.sh" "$EXP/results/ssa" 2>/dev/null || true

for bench in "${BENCHES[@]}"; do
  rm -f "$EXP"/work/${bench}_shard_*.txt
  case $bench in
    sbv) LIST="$EXP/sbv_sample_100.txt" ;;
    tb2) LIST="$EXP/tb2_all_89.txt" ;;
  esac
  awk -v k="$SHARDS_PER_BENCH" -v pre="$EXP/work/${bench}_shard_" \
    '{f=sprintf("%s%02d.txt", pre, NR%k); print > f}' "$LIST"
done

i=0
for bench in "${BENCHES[@]}"; do
  for f in "$EXP"/work/${bench}_shard_*.txt; do
    node="${NODES[$((i % ${#NODES[@]}))]}"
    part=docker; [ "$node" = ruby ] && part=dept
    name="ssa-${bench}-$(basename "$f" .txt | sed 's/.*_//')"
    jid=$(sbatch --parsable -J "$name" -p "$part" -w "$node" \
      --cpus-per-task=4 --mem=12G -t 24:00:00 \
      -o "$EXP/results/ssa/logs/${name}-%j.log" \
      --wrap "bash $EXP/scripts/run_shard.sh $bench $f")
    echo "submitted $name -> job $jid on $node"
    i=$((i+1))
  done
done
echo "all shards submitted: $i jobs"
