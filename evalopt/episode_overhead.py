"""Per-episode fixed cost in one simulator process (eval phase only, as with replayed expert checks).

Episodes (seed list) run back to back like the evaluator: setup_demo -> first head frame -> N moving
steps (take_action + head frame) -> close_env. sapien's clear_cache is called after the episode index
given by --clear_after (the official loop clears every clear_cache_freq=5 episodes).
Reports per episode: setup_demo, first frame, median later frame, close; and hashes of every frame so
the same seed before and after a cache clear can be compared.
usage: python episode_overhead.py --out f.json [--task ... --seeds 100000 100003 100000 --clear_after 1]
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
ap.add_argument("--task", default="place_container_plate")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seeds", type=int, nargs="+", default=[100000, 100003, 100000, 100003])
ap.add_argument("--clear_after", type=int, default=1)
ap.add_argument("--clear_every", type=int, default=0, help="if >0, clear after every k-th episode instead")
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)


def frame(head):
    t = time.perf_counter()
    env._update_render()
    head.take_picture()
    img = head.get_picture("Color")
    return (time.perf_counter() - t) * 1e3, hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:12]


res = []
for i, seed in enumerate(cli.seeds):
    t = time.perf_counter()
    env.setup_demo(now_ep_num=i, seed=seed, is_test=True, **args)
    setup = time.perf_counter() - t
    head = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"]
    tf0 = time.time()
    f0, h0 = frame(head)
    tf1 = time.time()
    base = np.asarray(env.get_obs()["joint_action"]["vector"], dtype=np.float32)
    m = np.ones_like(base)
    m[[6, 13]] = 0.0
    ft, hs = [], [h0]
    for k in range(cli.steps):
        env.take_action(base + 0.05 * np.sin(0.3 * k + np.arange(base.size)) * m, action_type="qpos")
        dt, hh = frame(head)
        ft.append(dt)
        hs.append(hh)
    t = time.perf_counter()
    do_clear = ((i + 1) % cli.clear_every == 0) if cli.clear_every else (i == cli.clear_after)
    env.close_env(clear_cache=do_clear)
    close = time.perf_counter() - t
    res.append({"i": i, "seed": seed, "setup_s": round(setup, 3), "first_frame_ms": round(f0, 1),
                "later_frame_ms_median": round(sorted(ft)[len(ft) // 2], 1), "close_s": round(close, 3),
                "cleared_after": do_clear, "first_frame_window": [round(tf0, 3), round(tf1, 3)],
                "frames": hs})
    print(json.dumps({k: v for k, v in res[-1].items() if k != "frames"}), flush=True)
by_seed = {}
for r in res:
    by_seed.setdefault(r["seed"], []).append(r["frames"])
out = {"episodes": res, "same_seed_frames_identical": {s: all(f == v[0] for f in v) for s, v in by_seed.items()}}
Path(cli.out).write_text(json.dumps(out, indent=1))
print(json.dumps(out["same_seed_frames_identical"]))
