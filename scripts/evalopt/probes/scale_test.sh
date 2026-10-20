#!/usr/bin/env bash
# Render concurrency scaling: K identical sim processes on one GPU, started together.
# usage: scale_test.sh <gpu> <out_dir> K1 K2 ...
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
GPU=$1; OUT=$2; shift 2
source "$SIM_ACTIVATE"
export PYTHONWARNINGS=ignore
mkdir -p "$OUT"
for K in "$@"; do
  D="$OUT/k$K"; mkdir -p "$D"; rm -f "$D"/flag*
  nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 500 > "$D/gpu.csv" &
  SMI=$!
  for i in $(seq 1 "$K"); do
    CUDA_VISIBLE_DEVICES=$GPU timeout 1200 python "$EVALOPT_PY/render_scale.py" \
      --out "$D/p$i.json" --start_flag "$D/flag" > "$D/p$i.log" 2>&1 &
  done
  until [ "$(ls "$D"/flag.ready.* 2>/dev/null | wc -l)" -ge "$K" ]; do sleep 1; done
  touch "$D/flag"
  date +%s.%N > "$D/t_start"
  wait $(jobs -p | grep -v "^$SMI$") 2>/dev/null
  date +%s.%N > "$D/t_end"
  kill $SMI 2>/dev/null
  echo "K=$K done"
done
