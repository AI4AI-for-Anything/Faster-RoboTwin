#!/usr/bin/env bash
# Shared start-up: ONE policy server and ONE simulator process run a whole list of jobs.
# The server is told to reset itself at every job start (EVALOPT_SERVER_NEW_JOB=1).
# NO_SERVER=1: do not start a server, use the (multi-session) one already on <port>.
# CPROF=1 runs the simulator worker under cProfile (cprof.out in <worker_out>).
# jobs.json comes from evalopt/make_jobs.py.
# usage: run_resident.sh <gpu> <port> <jobs.json> <worker_out> [ENV=VAL ...]
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
evalopt_require_ckpt
GPU=$1; PORT=$2; JOBS=$3; OUT=$4; shift 4
mkdir -p "$OUT"
if [ "${NO_SERVER:-0}" = 1 ]; then
  # a shared (multi-session) server is already listening on $PORT
  port_ours "$PORT" || { echo "no server of ours on $PORT" >&2; exit 5; }
  SP=""
else
  port_free "$PORT" || { echo "port $PORT busy" >&2; exit 4; }
  PROF_SERVER_OUT="$OUT/server_prof.jsonl" setsid "$EVALOPT_SH/serve_policy.sh" "$GPU" "$PORT" "$OUT/server.log" &
  SP=$!
  for i in $(seq 1 120); do grep -q model_init "$OUT/server_prof.jsonl" 2>/dev/null && port_ours "$PORT" && break; sleep 2; done
fi
(
  source "$SIM_ACTIVATE"
  export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning EVALOPT_SERVER_NEW_JOB=1
  for kv in "$@"; do export "$kv"; done
  cd "$ROBOTWIN_ROOT"
  date +%s.%N > "$OUT/t_start"
  PRE=(); [ "${CPROF:-0}" = 1 ] && PRE=(-m cProfile -o "$OUT/cprof.out")
  python "${PRE[@]}" "$EVALOPT_PY/prof_eval.py" --jobs "$JOBS" > "$OUT/worker.log" 2> "$OUT/worker.err"
  echo $? > "$OUT/exit_code"
  date +%s.%N > "$OUT/t_end"
)
[ -n "$SP" ] && { kill -TERM -- -"$SP" 2>/dev/null || kill "$SP"; }
