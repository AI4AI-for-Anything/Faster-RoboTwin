#!/usr/bin/env bash
# K concurrent eval jobs whose simulators share one GPU (each job: own fresh policy server on a
# server GPU, own simulator on <sim_gpu>). Measures how job throughput scales once rendering is cut.
# usage: [PORT_BASE=19000] scale_jobs.sh <task> <config> <n> <sim_gpu> <server_gpus_csv> <out> <K...>   (env: EVALOPT_* for the arms)
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
TASK=$1; CFG=$2; N=$3; SG=$4; IFS=, read -ra SVG <<< "$5"; OUT=$6; shift 6
mkdir -p "$OUT"
for K in "$@"; do
  D="$OUT/k$K"; mkdir -p "$D"; SP=()
  for i in $(seq 1 "$K"); do
    g=${SVG[$(( (i-1) % ${#SVG[@]} ))]}; port=$((${PORT_BASE:-19000} + i))
    mkdir -p "$D/j$i"
    port_free "$port" || { echo "port $port busy" >&2; exit 4; }
    PROF_SERVER_OUT="$D/j$i/server_prof.jsonl" setsid "$EVALOPT_SH/serve_policy.sh" "$g" "$port" "$D/j$i/server.log" &
    SP+=($!)
  done
  for i in $(seq 1 "$K"); do
    port=$((${PORT_BASE:-19000} + i))
    for t in $(seq 1 120); do grep -q model_init "$D/j$i/server_prof.jsonl" 2>/dev/null && port_ours "$port" && break; sleep 2; done
  done
  date +%s.%N > "$D/t_start"
  for i in $(seq 1 "$K"); do
    (cd "$ROBOTWIN_ROOT" && "$EVALOPT_SH/run_case.sh" "$TASK" "$CFG" "$N" "$D/j$i" "$SG" $((${PORT_BASE:-19000} + i))) &
  done
  wait $(jobs -p | grep -v -E "^($(IFS='|'; echo "${SP[*]}"))$") 2>/dev/null
  date +%s.%N > "$D/t_end"
  for p in "${SP[@]}"; do kill -TERM -- -"$p" 2>/dev/null || kill "$p"; done
  sleep 5
  echo "K=$K done"
done
