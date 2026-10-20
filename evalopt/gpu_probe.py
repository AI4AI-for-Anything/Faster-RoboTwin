"""GPU-share probe: render the head camera of one fixed scene in a tight loop for --seconds and log
(timestamp, frame ms) for every frame. Run it next to another process on the same GPU; frames that
slow down while the other process does X show how much of X occupies the GPU.
usage: python gpu_probe.py --out f.jsonl --seconds 120
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_sched  # noqa: E402,F401  (EVALOPT_CUDA_BLOCKING_SYNC, before any CUDA init)
sys.path.insert(0, str(ROBOTWIN / "scripts"))

import eval_policy_xpolicylab as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="place_container_plate")
ap.add_argument("--seed", type=int, default=100000)
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seconds", type=float, default=120)
ap.add_argument("--ready", default=None)
ap.add_argument("--go", default=None, help="after signalling ready, wait for this file before starting")
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "probe"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)
env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
head = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"]
for _ in range(5):
    env._update_render()
    head.take_picture()
    head.get_picture("Color")
if cli.ready:
    Path(cli.ready).touch()
if cli.go:
    while not Path(cli.go).exists():
        time.sleep(0.05)
f = open(cli.out, "w")
t_end = time.time() + cli.seconds
while time.time() < t_end:
    t = time.perf_counter()
    env._update_render()
    head.take_picture()
    head.get_picture("Color")
    f.write(json.dumps({"t": round(time.time(), 3), "ms": round((time.perf_counter() - t) * 1e3, 2)}) + "\n")
f.close()
