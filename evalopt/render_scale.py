"""One process of a render-concurrency test: build the eval scene for (task, seed), replay a fixed
synthetic qpos sequence with the official per-step get_obs(), report per-step get_obs wall time and
hashes of every rendered image (to check concurrency does not change pixels).
usage: python render_scale.py --out f.json [--task adjust_bottle --config demo_clean --seed 100001 --steps 40]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROBOTWIN / "scripts"))

import numpy as np  # noqa: E402

import eval_policy_xpolicylab as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="adjust_bottle")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seed", type=int, default=100001)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--start_flag", default=None)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)
env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
obs = env.get_obs()
base = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
arm = np.ones_like(base)
arm[[6, 13]] = 0.0
ACTIONS = [base + 0.08 * np.sin(0.25 * t + np.arange(base.size)) * arm for t in range(cli.steps)]

if cli.start_flag:
    Path(cli.start_flag + f".ready.{os.getpid()}").touch()
    while not Path(cli.start_flag).exists():
        time.sleep(0.05)

seq = hashlib.sha256()
obs_ms, act_ms = [], []
t_all = time.perf_counter()
c_all = time.process_time()
for a in ACTIONS:
    t = time.perf_counter()
    env.take_action(a.astype(np.float32), action_type="qpos")
    act_ms.append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter()
    obs = env.get_obs()
    obs_ms.append((time.perf_counter() - t) * 1e3)
    for k in sorted(obs["observation"]):
        if "rgb" in obs["observation"][k]:
            seq.update(np.ascontiguousarray(obs["observation"][k]["rgb"]).tobytes())
res = {"pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
       "wall_s": round(time.perf_counter() - t_all, 3), "cpu_s": round(time.process_time() - c_all, 3),
       "get_obs_ms_median": round(float(np.median(obs_ms)), 1),
       "take_action_ms_median": round(float(np.median(act_ms)), 1),
       "image_seq_sha": seq.hexdigest()[:16]}
env.close_env()
Path(cli.out).write_text(json.dumps(res))
print(json.dumps(res))
