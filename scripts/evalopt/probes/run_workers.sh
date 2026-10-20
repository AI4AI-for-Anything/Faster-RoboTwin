#!/usr/bin/env bash
# Deployment shape: K resident workers (each = run_resident.sh: own resident policy server + own
# resident simulator process) on ONE GPU, started together. One jobs.json per worker.
# Also samples GPU util / SM clock / power and per-process CPU every second.
# usage: run_workers.sh <gpu> <port_base> <out> <jobs1.json> [<jobs2.json> ...]   (env passed through)
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
GPU=$1; PB=$2; OUT=$3; shift 3
mkdir -p "$OUT"
nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader -lms 1000 > "$OUT/gpu.csv" &
SMI=$!
date +%s.%N > "$OUT/t_start"
i=0; PIDS=()
for J in "$@"; do
  i=$((i + 1))
  "$EVALOPT_SH/run_resident.sh" "$GPU" $((PB + i)) "$J" "$OUT/w$i" > "$OUT/w$i.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
date +%s.%N > "$OUT/t_end"
kill $SMI 2>/dev/null
echo done > "$OUT/DONE"
