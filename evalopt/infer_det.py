"""Is official Pi_05 inference bitwise reproducible? Same obs, same noise / same rng chain.

Loads the policy exactly like XPolicyLab Pi_05 Model.get_model (official config + checkpoint
norm stats), feeds a fixed deterministic observation and prints output hashes for:
  fixed_noise  - infer(obs, noise=N) 3 times (no RNG involved)
  rng_chain    - fresh Policy rng (key(0)), infer(obs) 3 times
usage: PI05_DEPLOY_YML=... PI05_CKPT=... python infer_det.py <out.json>   (policy env)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import yaml

import XPolicyLab.policy.Pi_05.model as M

cfg = yaml.safe_load(open(os.environ["PI05_DEPLOY_YML"]))
cfg["model_path"] = cfg["ckpt_name"] = os.environ["PI05_CKPT"]  # step directory 29999
t0 = time.perf_counter()
model = M.Model(cfg)
load_s = time.perf_counter() - t0
policy = model.policy

rs = np.random.RandomState(1234)
obs = {
    "state": rs.uniform(-0.5, 0.5, size=14).astype(np.float32),
    "images": {k: rs.randint(0, 255, size=(3, 240, 320), dtype=np.uint8)
               for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
    "prompt": "Lift the medium-sized green bottle head-up from the table",
}


def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


out = {"load_s": round(load_s, 2)}
noise = np.random.RandomState(7).standard_normal((50, 32)).astype(np.float32)
res, times = [], []
for _ in range(3):
    t = time.perf_counter()
    res.append(h(policy.infer(obs, noise=noise)["actions"]))
    times.append(round(time.perf_counter() - t, 4))
out["fixed_noise"] = res
out["fixed_noise_s"] = times
res = []
import jax  # noqa: E402

policy._rng = jax.random.key(0)
for _ in range(3):
    res.append(h(policy.infer(obs)["actions"]))
out["rng_chain"] = res
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out))
