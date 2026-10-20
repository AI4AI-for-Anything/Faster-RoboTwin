"""Render cost per camera and render statelessness checks on a real RoboTwin task scene.

Builds the scene exactly like the official evaluator (load_task_args + setup_demo with the
eval seed), then:
  timing  - take_picture/get_picture per camera, wall and process CPU time
  regimes - same synthetic joint-action sequence under different render schedules, each from a
            fresh setup_demo(seed); compares final images and joint state bitwise:
            A  get_obs() before every action (official schedule)
            A2 repeat of A (repeatability inside one process)
            B  no renders between first and last get_obs()
            C  between boundaries only _update_render() + head camera (what an eval video needs)
            D  like A, but the front camera is never rendered
usage: python render_bench.py --task adjust_bottle --config demo_clean --seed 100001 --out f.json
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
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)


def setup():
    env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)


def cams():
    c = env.cameras
    out = {"left_camera": c.left_camera, "right_camera": c.right_camera}
    out.update(dict(zip(c.static_camera_name, c.static_camera_list)))
    return out


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def snapshot():
    obs = env.get_obs()
    imgs = {k: h(v["rgb"]) for k, v in obs["observation"].items() if "rgb" in v}
    return {"images": imgs, "qpos": h(obs["joint_action"]["vector"])}


res = {"task": cli.task, "config": cli.config, "seed": cli.seed}
t0 = time.perf_counter()
setup()
res["first_setup_s"] = round(time.perf_counter() - t0, 3)
res["cameras"] = {k: [int(v.get_width()), int(v.get_height())] for k, v in cams().items()}

# ---- timing ----
timing = {}
for name, cam in cams().items():
    w, p = [], []
    for _ in range(6):
        env._update_render()
        a, b = time.perf_counter(), time.process_time()
        cam.take_picture()
        cam.get_picture("Color")
        w.append(time.perf_counter() - a)
        p.append(time.process_time() - b)
    timing[name] = {"wall_ms": [round(x * 1e3, 1) for x in w], "cpu_ms": [round(x * 1e3, 1) for x in p]}
w = []
for _ in range(4):
    a = time.perf_counter()
    env.get_obs()
    w.append(time.perf_counter() - a)
timing["get_obs_all"] = {"wall_ms": [round(x * 1e3, 1) for x in w]}
res["timing"] = timing

# repeat render at identical state
env._update_render()
r1 = {k: h(v.get_picture("Color")) for k, v in cams().items() if not v.take_picture()}
r2 = {k: h(v.get_picture("Color")) for k, v in cams().items() if not v.take_picture()}
res["same_state_rerender_identical"] = r1 == r2

# ---- regimes ----
setup()
base = np.asarray(env.get_obs()["joint_action"]["vector"], dtype=np.float64)
arm = np.ones_like(base)
arm[[6, 13]] = 0.0
ACTIONS = [base + 0.08 * np.sin(0.25 * t + np.arange(base.size)) * arm for t in range(cli.steps)]


def run(regime):
    setup()
    head = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"]
    front_names = [n for n in env.cameras.static_camera_name if n != "head_camera"]
    if regime == "D":
        keep = [(c, n) for c, n in zip(env.cameras.static_camera_list, env.cameras.static_camera_name)
                if n == "head_camera"]
        saved = (env.cameras.static_camera_list, env.cameras.static_camera_name)
        env.cameras.static_camera_list = [c for c, _ in keep]
        env.cameras.static_camera_name = [n for _, n in keep]
    t0 = time.perf_counter()
    env.get_obs()
    for t, a in enumerate(ACTIONS):
        env.take_action(a.astype(np.float32), action_type="qpos")
        if t + 1 == len(ACTIONS):
            break
        if regime in ("A", "A2", "D"):
            env.get_obs()
        elif regime == "C":
            env._update_render()
            head.take_picture()
            head.get_picture("Color")
    snap = snapshot()
    snap["wall_s"] = round(time.perf_counter() - t0, 3)
    snap["take_action_cnt"] = env.take_action_cnt
    if regime == "D":
        env.cameras.static_camera_list, env.cameras.static_camera_name = saved
    snap["skipped_cams"] = front_names if regime == "D" else []
    env.close_env()
    return snap


res["regimes"] = {r: run(r) for r in ("A", "A2", "B", "C", "D")}
A = res["regimes"]["A"]
res["compare_to_A"] = {
    r: {"qpos_equal": v["qpos"] == A["qpos"],
        "images_equal": {k: v["images"].get(k) == A["images"][k] for k in A["images"] if k in v["images"]}}
    for r, v in res["regimes"].items() if r != "A"}
Path(cli.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res["compare_to_A"], indent=1))
print(json.dumps(res["timing"], indent=1))
