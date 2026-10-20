"""Does one GPU render frames from two scenes faster when they are submitted from ONE process
(one Vulkan device, two threads) than from two processes (time-sliced contexts)? And are the
pixels identical to rendering the same scene alone?

Builds TWO task environments in this process (seed A, seed B), then for N frames:
  serial   : env A head take_picture+get_picture, then env B
  threads  : the two cameras rendered at the same time from two Python threads
Reports wall per frame-pair and whether every image equals the serial one.
usage: python render_concurrency.py --out f.json [--task place_container_plate --seeds 100000 100003]
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROBOTWIN / "scripts"))

import numpy as np  # noqa: E402

import eval_policy_xpolicylab as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="place_container_plate")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seeds", type=int, nargs=2, default=[100000, 100003])
ap.add_argument("--frames", type=int, default=12)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
envs, heads = [], []
for s in cli.seeds:
    env = E.class_decorator(cli.task)
    env.setup_demo(now_ep_num=0, seed=s, is_test=True, **args)
    envs.append(env)
    heads.append(dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"])


def render(i, out):
    envs[i]._update_render()
    heads[i].take_picture()
    out[i] = hashlib.sha256(np.ascontiguousarray(heads[i].get_picture("Color")).tobytes()).hexdigest()[:16]


for _ in range(2):
    render(0, {}), render(1, {})
ser, thr, same = [], [], []
ref = None
for _ in range(cli.frames):
    o = {}
    t = time.perf_counter()
    render(0, o)
    render(1, o)
    ser.append(time.perf_counter() - t)
    ref = dict(o)
    o2 = {}
    th = [threading.Thread(target=render, args=(i, o2)) for i in (0, 1)]
    t = time.perf_counter()
    for x in th:
        x.start()
    for x in th:
        x.join()
    thr.append(time.perf_counter() - t)
    same.append(o2 == ref)
res = {"serial_pair_ms_median": round(sorted(ser)[len(ser) // 2] * 1e3, 1),
       "threaded_pair_ms_median": round(sorted(thr)[len(thr) // 2] * 1e3, 1),
       "threaded_pixels_identical": all(same), "frames": cli.frames}
Path(cli.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res))
