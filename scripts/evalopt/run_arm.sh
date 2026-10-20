#!/usr/bin/env bash
# One profiled arm with its own fresh policy server (server + simulator on <gpu>).
# CPROF=1 additionally runs the simulator process under cProfile (cprof.out in <out>).
# usage: run_arm.sh <task> <config> <n> <gpu> <port> <out> [ENV=VAL ...]
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
evalopt_require_ckpt
TASK=$1; CFG=$2; N=$3; GPU=$4; PORT=$5; OUT=$6; shift 6
mkdir -p "$OUT"
port_free "$PORT" || { echo "port $PORT busy" >&2; exit 4; }
PROF_SERVER_OUT="$OUT/server_prof.jsonl" setsid "$EVALOPT_SH/serve_policy.sh" "$GPU" "$PORT" "$OUT/server.log" &
SP=$!
for i in $(seq 1 120); do grep -q model_init "$OUT/server_prof.jsonl" 2>/dev/null && port_ours "$PORT" && break; sleep 2; done
(
  source "$SIM_ACTIVATE"
  export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning
  for kv in "$@"; do export "$kv"; done
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -lms 1000 > "$OUT/gpu.csv" &
  SMI=$!
  PRE=(); [ "${CPROF:-0}" = 1 ] && PRE=(-m cProfile -o "$OUT/cprof.out")
  cd "$ROBOTWIN_ROOT"
  python "${PRE[@]}" "$EVALOPT_PY/prof_eval.py" --prof_out "$OUT" -- \
    --bench_name RoboTwin --task_name "$TASK" --env_cfg_type arx_x5 --policy_name Pi_05 \
    --host 127.0.0.1 --port "$PORT" --protocol ws --device_id "$GPU" \
    --seed 0 --test_num "$N" --task_config "$CFG" --expert_check true --eval_batch false \
    --additional_info "ckpt_name=$PI05_CKPT,action_type=joint" \
    > "$OUT/eval.log" 2> "$OUT/eval.err"
  echo $? > "$OUT/exit_code"
  kill $SMI 2>/dev/null
)
kill -TERM -- -"$SP" 2>/dev/null || kill "$SP"
