# Paths shared by the evalopt launchers. Every value can be overridden from the environment.
# Sourced by the scripts in this directory; not meant to be executed.

EVALOPT_SH=${EVALOPT_SH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
export EVALOPT_SH
# RoboTwin checkout (this repository) and the evalopt Python tools inside it
export ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$(cd -- "$EVALOPT_SH/../.." && pwd)}
export EVALOPT_PY=${EVALOPT_PY:-$ROBOTWIN_ROOT/evalopt}
# Script that activates the simulator environment (Python env + Vulkan/EGL variables);
# start from activate-sim.example.sh
export SIM_ACTIVATE=${SIM_ACTIVATE:-$EVALOPT_SH/activate-sim.sh}
# Policy environment created by XPolicyLab/policy/Pi_05/install.sh
export POLICY_VENV=${POLICY_VENV:-$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05/openpi/.venv}
# Official RoboTwin π0.5 co-train checkpoint: the step directory .../RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999
export PI05_CKPT=${PI05_CKPT:-}
# Copy of the upstream Pi_05 deploy.yml with the released step and normalization selected
export PI05_DEPLOY_YML=${PI05_DEPLOY_YML:-$EVALOPT_SH/configs/official-pi05-deploy.yml}
# Default log directory for servers started without an explicit log path
export EVALOPT_LOG_DIR=${EVALOPT_LOG_DIR:-$ROBOTWIN_ROOT/evalopt_runs/logs}

evalopt_require_ckpt() {
  [[ -n "$PI05_CKPT" && -d "$PI05_CKPT" ]] || {
    echo "Set PI05_CKPT to the official checkpoint step directory (.../RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999)." >&2
    exit 2
  }
}

# Other services may listen on local ports of a shared machine: refuse a busy port, and only treat a
# server as ready when the listener belongs to one of our own processes (ss shows pid= only for those).
port_free() { ! ss -ltn | grep -q ":$1 "; }
port_ours() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -q "pid="; }
