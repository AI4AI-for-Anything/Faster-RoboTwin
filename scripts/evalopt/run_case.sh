#!/usr/bin/env bash
# One profiled official-protocol case against a running policy server.
# usage: run_case.sh <task> <demo_clean|demo_randomized> <test_num> <out_dir> <sim_gpu> <port>
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
evalopt_require_ckpt
TASK=$1; CFG=$2; N=$3; OUT=$4; GPU=$5; PORT=$6
mkdir -p "$OUT"
source "$SIM_ACTIVATE"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -lms 1000 > "$OUT/gpu.csv" &
SMI=$!
cd "$ROBOTWIN_ROOT"
python "$EVALOPT_PY/prof_eval.py" --prof_out "$OUT" -- \
  --bench_name RoboTwin --task_name "$TASK" --env_cfg_type arx_x5 --policy_name Pi_05 \
  --host 127.0.0.1 --port "$PORT" --protocol ws --device_id "$GPU" \
  --seed 0 --test_num "$N" --task_config "$CFG" --expert_check true --eval_batch false \
  --additional_info "ckpt_name=$PI05_CKPT,action_type=joint" \
  > "$OUT/eval.log" 2> "$OUT/eval.err"
echo $? > "$OUT/exit_code"
kill $SMI 2>/dev/null
