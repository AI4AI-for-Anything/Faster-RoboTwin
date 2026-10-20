"""Does the expert check (play_once, cuRobo planning) leave the following eval episode reproducible?

Per trial, with the official evaluator's arguments:
  setup_demo(seed) -> play_once -> close_env      (expert check, as in eval_remote_policy)
  setup_demo(seed) -> get_obs -> 30 synthetic qpos actions with get_obs each step
Records the planned joint paths, success flags, the first eval obs and the final obs.
Also one trial with no expert check before the eval setup.
usage: python expert_det.py --task adjust_bottle --seed 100001 --out f.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROBOTWIN / "scripts"))

import numpy as np  # noqa: E402

import eval_policy_xpolicylab as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="adjust_bottle")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seed", type=int, default=100001)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)


def h_obs(obs):
    m = hashlib.sha256()
    for k in sorted(obs["observation"]):
        v = obs["observation"][k]
        if "rgb" in v:
            m.update(np.ascontiguousarray(v["rgb"]).tobytes())
    m.update(np.ascontiguousarray(obs["joint_action"]["vector"]).tobytes())
    return m.hexdigest()[:16]


def h_path(path):
    m = hashlib.sha256()
    for item in path or []:
        if isinstance(item, dict):
            for k in sorted(item):
                try:
                    m.update(np.ascontiguousarray(np.asarray(item[k], dtype=np.float64)).tobytes())
                except Exception:
                    m.update(str(item[k]).encode())
    return m.hexdigest()[:16]


ACTIONS = None


def eval_part():
    global ACTIONS
    env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
    obs = env.get_obs()
    first = h_obs(obs)
    if ACTIONS is None:
        base = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
        arm = np.ones_like(base)
        arm[[6, 13]] = 0.0
        ACTIONS = [base + 0.08 * np.sin(0.25 * t + np.arange(base.size)) * arm for t in range(30)]
    for a in ACTIONS:
        env.take_action(a.astype(np.float32), action_type="qpos")
        obs = env.get_obs()
    rec = {"first_obs": first, "final_obs": h_obs(obs)}
    env.close_env()
    return rec


trials = []
for trial in range(3):
    rec = {}
    if trial < 2:
        env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
        info = env.play_once()
        rec["expert"] = {"plan_success": bool(env.plan_success), "check_success": bool(env.check_success()),
                         "left_path": h_path(env.left_joint_path), "right_path": h_path(env.right_joint_path),
                         "info": json.dumps(info, sort_keys=True, default=str)[:200],
                         "end_obs": h_obs(env.get_obs())}
        env.close_env()
    else:
        rec["expert"] = None
    rec.update(eval_part())
    trials.append(rec)
Path(cli.out).write_text(json.dumps(trials, indent=1))
print(json.dumps(trials, indent=1))
