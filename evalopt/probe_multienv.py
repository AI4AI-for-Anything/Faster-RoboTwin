"""One process, N task environments (different seeds), head camera rendered round-robin from a
single thread for --seconds: GPU frame rate when N scenes share one Vulkan device/context.
Every frame is hashed; with --ref_hash_from a json of single-scene hashes the first frame of each
scene can be compared. Logs (t, ms, scene) per frame.
usage: python probe_multienv.py --n 4 --out f.jsonl [--task ... --config ... --seconds 120 --ready R --go G]
"""
import argparse
import hashlib
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

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="place_container_plate")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seeds", type=int, nargs="+", default=[100000, 100003, 100004, 100005, 100006, 100007, 100008, 100009,
                                                         100010, 100011, 100012, 100013, 100014, 100015])
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--seconds", type=float, default=120)
ap.add_argument("--ready", default=None)
ap.add_argument("--go", default=None)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "probe"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
envs, heads, built = [], [], []
for s in cli.seeds:
    if len(envs) == cli.n:
        break
    env = E.class_decorator(cli.task)
    try:
        env.setup_demo(now_ep_num=0, seed=s, is_test=True, **args)
    except Exception:
        continue
    envs.append(env)
    built.append(s)
    heads.append(dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"])


def frame(i):
    envs[i]._update_render()
    heads[i].take_picture()
    return heads[i].get_picture("Color")


for i in range(len(envs)):
    for _ in range(3):
        frame(i)
if cli.ready:
    Path(cli.ready).touch()
if cli.go:
    while not Path(cli.go).exists():
        time.sleep(0.05)
f = open(cli.out, "w")
f.write(json.dumps({"seeds": built, "cuda_sched": cuda_sched.STATUS}) + "\n")
t_end = time.time() + cli.seconds
k = 0
while time.time() < t_end:
    i = k % len(envs)
    t = time.perf_counter()
    img = frame(i)
    f.write(json.dumps({"t": round(time.time(), 3), "ms": round((time.perf_counter() - t) * 1e3, 2), "i": i,
                        "h": hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:10]}) + "\n")
    k += 1
f.close()
