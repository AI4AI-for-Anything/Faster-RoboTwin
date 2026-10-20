#!/usr/bin/env bash
# Reference vs candidate on one case, each with a freshly started policy server (the JAX rng chain
# restarts at key(0), as for one official task job). Server and simulator share GPU <gpu> like the
# official eval.sh. Arms run in order ref, opt (and ref2 if REPEAT_REF=1). REF_ENV adds switches to
# both reference arms (for example EVALOPT_EXPERT_REPLAY=<pkl> so that both arms replay one record).
# Judge the result with: python evalopt/compare_arms.py <out_root>
# usage: run_pair.sh <task> <config> <n> <gpu> <port> <out_root> [opt_env...]
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
evalopt_require_ckpt
TASK=$1; CFG=$2; N=$3; GPU=$4; PORT=$5; OUT=$6; shift 6
OPT_ENV=("$@"); [ ${#OPT_ENV[@]} -eq 0 ] && OPT_ENV=(EVALOPT_SKIP_UNUSED_CAMS=1 EVALOPT_LAZY_OBS=1)
mkdir -p "$OUT"
arm() {  # name, env...
  local name=$1; shift
  local D="$OUT/$name"; mkdir -p "$D"
  port_free "$PORT" || { echo "port $PORT busy" >&2; exit 4; }
  PROF_SERVER_OUT="$D/server_prof.jsonl" setsid "$EVALOPT_SH/serve_policy.sh" "$GPU" "$PORT" "$D/server.log" &
  local SP=$!
  for i in $(seq 1 120); do grep -q model_init "$D/server_prof.jsonl" 2>/dev/null && port_ours "$PORT" && break; sleep 2; done
  (cd "$ROBOTWIN_ROOT" && env "$@" "$EVALOPT_SH/run_case.sh" "$TASK" "$CFG" "$N" "$D" "$GPU" "$PORT")
  kill -TERM -- -"$SP" 2>/dev/null || kill "$SP"
  wait "$SP" 2>/dev/null
  for i in $(seq 1 30); do port_ours "$PORT" || break; sleep 1; done
}
read -ra REF_ENV_A <<< "${REF_ENV:-EVALOPT_NONE=1}"
arm ref "${REF_ENV_A[@]}"
arm opt "${OPT_ENV[@]}"
[ "${REPEAT_REF:-0}" = 1 ] && arm ref2 "${REF_ENV_A[@]}"
echo done > "$OUT/DONE"
