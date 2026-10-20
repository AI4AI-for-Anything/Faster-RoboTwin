"""Official Pi_05 policy server with timing wrappers (no behaviour change).

Wraps Model.__init__/update_obs/get_action/reset and openpi Policy.infer; every wrapper calls the
original and returns its result. Adds one RPC, evalopt_new_job, used when a server is reused across
jobs (prof_eval.py EVALOPT_SERVER_NEW_JOB=1). Writes one JSON line per call to $PROF_SERVER_OUT.

usage: same args as XPolicyLab/setup_policy_server.py
"""
import functools
import json
import os
import sys
import time
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROBOTWIN / "XPolicyLab"))
T0 = time.perf_counter()
LOG = open(os.environ["PROF_SERVER_OUT"], "a", buffering=1)

import setup_policy_server as S  # noqa: E402
import XPolicyLab.policy.Pi_05.model as M  # noqa: E402
from openpi.policies import policy as P  # noqa: E402

LOG.write(json.dumps({"n": "import", "d": round(time.perf_counter() - T0, 3)}) + "\n")


def timed(owner, attr, name):
    fn = getattr(owner, attr)

    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            LOG.write(json.dumps({"n": name, "t": round(t0 - T0, 4),
                                  "d": round(time.perf_counter() - t0, 5)}) + "\n")

    setattr(owner, attr, w)


def evalopt_new_job(self):
    """Return a reused server to the state of a freshly started one: the openpi Policy's JAX
    random-number chain restarts at key(0) (Policy.__init__ default) and the observation is dropped."""
    import jax

    self.policy._rng = jax.random.key(0)
    self.reset()


M.Model.evalopt_new_job = evalopt_new_job
timed(M.Model, "evalopt_new_job", "evalopt_new_job")
timed(M.Model, "__init__", "model_init")
timed(M.Model, "update_obs", "update_obs")
timed(M.Model, "get_action", "get_action")
timed(M.Model, "reset", "reset")
timed(P.Policy, "infer", "policy_infer")
for attr in ("encode_obs", "stack_obs"):
    timed(M, attr, attr)

if __name__ == "__main__":
    S.main(S.parse_args_and_config())
