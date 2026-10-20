"""Official take_action vs take_action_opt (no per-substep render sync), same scene and actions.

From a fresh setup_demo(seed) each arm applies the same fixed qpos action sequence; after every
action it renders the head camera like the eval loop's mid-chunk step. Records per-step wall time of
take_action and of the head frame, and the hash of every head frame and of the joint state.
usage: python substep_bench.py --out f.json [--task ... --config ... --seed ... --steps 40]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import eval_policy_xpolicylab as E  # noqa: E402
import envs._base_task as BT  # noqa: E402
import opt_take_action  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="place_container_plate")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seed", type=int, default=100000)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)
OFFICIAL = BT.Base_Task.take_action


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def arm(fn):
    BT.Base_Task.take_action = fn
    env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
    head = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"]
    obs = env.get_obs()
    base = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    m = np.ones_like(base)
    m[[6, 13]] = 0.0
    act_ms, frame_ms, imgs, qs = [], [], [], []
    for t in range(cli.steps):
        a = time.perf_counter()
        env.take_action(base + 0.06 * np.sin(0.2 * t + np.arange(base.size)) * m, action_type="qpos")
        act_ms.append((time.perf_counter() - a) * 1e3)
        a = time.perf_counter()
        env._update_render()
        head.take_picture()
        img = head.get_picture("Color")
        frame_ms.append((time.perf_counter() - a) * 1e3)
        imgs.append(h(img))
        qs.append(h(np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState())))
    env.close_env()
    med = lambda x: round(sorted(x)[len(x) // 2], 2)
    return {"take_action_ms_median": med(act_ms), "head_frame_ms_median": med(frame_ms),
            "imgs": imgs, "qpos": qs, "crazy_light": bool(env.crazy_random_light)}


res = {"official": arm(OFFICIAL), "opt": arm(opt_take_action.take_action_opt), "official2": arm(OFFICIAL)}
for k in ("opt", "official2"):
    res[k + "_equal"] = {"imgs": res[k]["imgs"] == res["official"]["imgs"], "qpos": res[k]["qpos"] == res["official"]["qpos"]}
Path(cli.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("imgs", "qpos")} if isinstance(v, dict) else v)
                  for k, v in res.items()}))
