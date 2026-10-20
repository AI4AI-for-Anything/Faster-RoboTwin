#!/usr/bin/env bash
# Run probe_scale.sh configurations ONE AFTER ANOTHER (quiet host), each on <gpu>.
# usage: probe_series.sh <gpu> <task> <config> <out_root> <seconds> "K N" ["K N" ...]
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
GPU=$1; TASK=$2; CFG=$3; OUT=$4; SEC=$5; shift 5
mkdir -p "$OUT"
for kn in "$@"; do
  set -- $kn
  "$EVALOPT_SH/probes/probe_scale.sh" "$GPU" "$1" "$2" "$TASK" "$CFG" "$OUT/K$1_N$2" "$SEC" >> "$OUT/series.log" 2>&1
done
echo done > "$OUT/DONE"
