#!/usr/bin/env bash
# Official eval with no wrappers at all (plain server + scripts/eval_policy.sh), pinned autotune.
# Used to show the profiling wrappers are transparent: compare its episode videos (md5) and
# success line with a wrapped reference arm of the same case.
# Note: upstream scripts/eval_policy.sh activates a conda environment; with a venv-based simulator
# environment it needs that activation adapted first (see docs/evaluation.md).
# usage: run_plain.sh <task> <config> <n> <gpu> <port> <out>
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
evalopt_require_ckpt
TASK=$1; CFG=$2; N=$3; GPU=$4; PORT=$5; OUT=$6
mkdir -p "$OUT"
port_free "$PORT" || { echo "port $PORT busy" >&2; exit 4; }
setsid "$EVALOPT_SH/serve_policy.sh" "$GPU" "$PORT" "$OUT/server.log" &
SP=$!
for i in $(seq 1 120); do port_ours "$PORT" && break; sleep 2; done
(
  source "$SIM_ACTIVATE"
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ROBOTWIN_ROOT"
  bash scripts/eval_policy.sh --bench_name RoboTwin --task_name "$TASK" --env_cfg_type arx_x5 \
    --policy_name Pi_05 --host 127.0.0.1 --port "$PORT" --protocol ws --device_id "$GPU" \
    --seed 0 --test_num "$N" --task_config "$CFG" --expert_check true --eval_batch false \
    --additional_info "ckpt_name=$PI05_CKPT,action_type=joint" \
    > "$OUT/eval.log" 2> "$OUT/eval.err"
  echo $? > "$OUT/exit_code"
)
kill -TERM -- -"$SP" 2>/dev/null || kill "$SP"
