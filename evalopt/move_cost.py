"""Why does a head frame cost more GPU time right after the scene moved? (attribution only)

Builds one eval scene and measures, for each kind of change made before the frame, the wall time of
_update_render + take_picture + get_picture("Color") of the head camera (CUDA blocking sync on, so the
wall time is the GPU time the frame waited for):
  static      nothing changed since the previous frame
  step_only   one scene.step() with the arms holding still (bodies may sleep)
  pose_only   robot joints teleported (set_qpos, no physics) - only render-side transforms change
  substep1    one physics substep of a real arm motion
  action      a full take_action (official: ~87 substeps, each followed by _update_render)
  action_nosub a full take_action without the per-substep render sync (opt_take_action.py)
usage: python move_cost.py --out f.json [--task ... --config ... --seed ... --reps 15]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_sched  # noqa: E402,F401
sys.path.insert(0, str(ROBOTWIN / "scripts"))

import numpy as np  # noqa: E402

import eval_policy_xpolicylab as E  # noqa: E402
import envs._base_task as BT  # noqa: E402
import opt_take_action  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="place_container_plate")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seed", type=int, default=100000)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)
env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
head = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"]
base = np.asarray(env.get_obs()["joint_action"]["vector"], dtype=np.float32)
m = np.ones_like(base)
m[[6, 13]] = 0.0
OFFICIAL = BT.Base_Task.take_action


def frame():
    t = time.perf_counter()
    env._update_render()
    head.take_picture()
    head.get_picture("Color")
    return (time.perf_counter() - t) * 1e3


k = [0]


def target():
    k[0] += 1
    return base + 0.05 * np.sin(0.3 * k[0] + np.arange(base.size)) * m


def do_static():
    pass


def do_step_only():
    env.scene.step()


def do_pose_only():
    q = env.robot.left_entity.get_qpos()
    env.robot.left_entity.set_qpos(q + 0.01 * np.sin(0.3 * k[0] + np.arange(q.size)))
    k[0] += 1


def do_substep1():
    a = target()
    env.robot.set_arm_joints(a[:6], np.zeros(6), "left")
    env.scene.step()


def do_action():
    BT.Base_Task.take_action = OFFICIAL
    env.take_action(target(), action_type="qpos")


def do_action_nosub():
    BT.Base_Task.take_action = opt_take_action.take_action_opt
    env.take_action(target(), action_type="qpos")


kinds = {"static": do_static, "step_only": do_step_only, "pose_only": do_pose_only, "substep1": do_substep1,
         "action": do_action, "action_nosub": do_action_nosub}
for _ in range(5):
    frame()
res = {}
for rnd in range(2):  # two interleaved rounds to see drift
    for name, fn in kinds.items():
        ms = []
        for _ in range(cli.reps):
            fn()
            ms.append(frame())
        res.setdefault(name, []).extend(ms)
out = {name: {"median_ms": round(sorted(v)[len(v) // 2], 2), "min_ms": round(min(v), 2), "n": len(v)}
       for name, v in res.items()}
out["cuda_sched"] = cuda_sched.STATUS
Path(cli.out).write_text(json.dumps(out, indent=1))
print(json.dumps(out))
