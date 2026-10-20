#!/usr/bin/env bash
# Deployment shape with ONE multi-session policy server per GPU: start it, then K resident simulator
# workers on the same GPU (run_resident.sh NO_SERVER=1), one jobs.json each, all using <port>.
# Samples GPU util / clock / power every second. Switches are passed through the environment
# (source scripts/evalopt/configs/faster.env first).
# usage: run_ms_workers.sh <gpu> <port> <out> <jobs1.json> [<jobs2.json> ...]
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
evalopt_require_ckpt
GPU=$1; PORT=$2; OUT=$3; shift 3
mkdir -p "$OUT"
port_free "$PORT" || { echo "port $PORT busy" >&2; exit 4; }
EVALOPT_MULTISESSION=1 PROF_SERVER_OUT="$OUT/server_prof.jsonl" setsid "$EVALOPT_SH/serve_policy.sh" "$GPU" "$PORT" "$OUT/server.log" &
SP=$!
for i in $(seq 1 120); do grep -q model_init "$OUT/server_prof.jsonl" 2>/dev/null && port_ours "$PORT" && break; sleep 2; done
nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader -lms 1000 > "$OUT/gpu.csv" &
SMI=$!
date +%s.%N > "$OUT/t_start"
i=0; PIDS=()
for J in "$@"; do
  i=$((i + 1))
  NO_SERVER=1 "$EVALOPT_SH/run_resident.sh" "$GPU" "$PORT" "$J" "$OUT/w$i" > "$OUT/w$i.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
date +%s.%N > "$OUT/t_end"
kill $SMI 2>/dev/null
kill -TERM -- -"$SP" 2>/dev/null || kill "$SP"
echo done > "$OUT/DONE"
