#!/usr/bin/env bash
# Official Pi_05 policy server (upstream entry; with PROF_SERVER_OUT set, the timing wrapper
# prof_server.py, which only adds clocks). XLA autotune results are pinned to one file recorded
# on an idle GPU (EVALOPT_AUTOTUNE, default below; set it to an empty string to disable): without it,
# kernel choice - and so the actions - depend on GPU load at server start-up. EVALOPT_MEM_FRACTION
# overrides the official 0.3 JAX memory share (allocator pool size only). EVALOPT_MULTISESSION=1 starts
# multisession_server.py (one server, one session per eval job).
# usage: serve_policy.sh <gpu> <port> [<log>]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
evalopt_require_ckpt
GPU=$1; PORT=$2; LOG=${3:-$EVALOPT_LOG_DIR/server_${PORT}.log}
mkdir -p "$(dirname -- "$LOG")"
AT=${EVALOPT_AUTOTUNE-$EVALOPT_PY/autotune/pi05_h20_idle.textproto}
source "$POLICY_VENV/bin/activate"
export PYTHONPATH="$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/XPolicyLab:$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05/openpi/src"
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=${EVALOPT_MEM_FRACTION:-0.3}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
[[ -n "$AT" ]] && export XLA_FLAGS="--xla_gpu_load_autotune_results_from=$AT${XLA_FLAGS:+ $XLA_FLAGS}"
cd "$ROBOTWIN_ROOT"
if [[ "${EVALOPT_MULTISESSION:-0}" == 1 ]]; then
  ENTRY="$EVALOPT_PY/multisession_server.py"
  export PROF_SERVER_OUT=${PROF_SERVER_OUT:-$LOG.prof.jsonl}
elif [[ -n "${PROF_SERVER_OUT:-}" ]]; then
  ENTRY="$EVALOPT_PY/prof_server.py"
else
  ENTRY=XPolicyLab/setup_policy_server.py
fi
exec python "$ENTRY" --config_path "$PI05_DEPLOY_YML" \
  --overrides port=$PORT model_path=$PI05_CKPT ckpt_name=$PI05_CKPT >"$LOG" 2>&1
