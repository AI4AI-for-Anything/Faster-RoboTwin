"""Where does one head-camera frame go? (attribution only; numbers from non-official settings are
never used for evaluation.)

Builds the eval scene for (task, seed) exactly like the evaluator. RENDER_VARIANT overrides the global
ray-tracing settings that setup_scene() applies (official: rt shader, 32 spp, depth 8, oidn):
  official | nodenoise | spp1 | raster
Per head-camera frame it times _update_render, take_picture and get_picture("Color") separately, wall
and process CPU, on (a) an unchanged scene and (b) a scene that moved one control step (take_action
with a fixed small joint offset), which forces acceleration-structure / pose updates.
usage: RENDER_VARIANT=official python render_internals.py --out f.json [--task ... --seed ...]
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

import numpy as np  # noqa: E402
import sapien  # noqa: E402

VARIANT = os.environ.get("RENDER_VARIANT", "official")
_orig = {k: getattr(sapien.render, k) for k in ("set_camera_shader_dir", "set_ray_tracing_samples_per_pixel",
                                                 "set_ray_tracing_denoiser")}
if VARIANT == "nodenoise":
    sapien.render.set_ray_tracing_denoiser = lambda name: _orig["set_ray_tracing_denoiser"]("none")
elif VARIANT == "spp1":
    sapien.render.set_ray_tracing_samples_per_pixel = lambda n: _orig["set_ray_tracing_samples_per_pixel"](1)
elif VARIANT == "raster":
    sapien.render.set_camera_shader_dir = lambda d: _orig["set_camera_shader_dir"]("default")

import eval_policy_xpolicylab as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="place_container_plate")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seed", type=int, default=100000)
ap.add_argument("--frames", type=int, default=25)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)
t0 = time.perf_counter()
env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)
setup_s = time.perf_counter() - t0
head = dict(zip(env.cameras.static_camera_name, env.cameras.static_camera_list))["head_camera"]
obs = env.get_obs()
base = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
arm = np.ones_like(base)
arm[[6, 13]] = 0.0


import hashlib  # noqa: E402

HASHES = []


def frame():
    r = {}
    for name, fn in (("update_render", env._update_render), ("take_picture", head.take_picture),
                     ("get_picture", lambda: HASHES.append(hashlib.sha256(head.get_picture("Color").tobytes()).hexdigest()[:12]))):
        a, b = time.perf_counter(), time.process_time()
        fn()
        r[name] = ((time.perf_counter() - a) * 1e3, (time.process_time() - b) * 1e3)
    return r


def summarize(rows):
    out = {}
    for k in rows[0]:
        w = sorted(x[k][0] for x in rows)
        c = sorted(x[k][1] for x in rows)
        out[k] = {"wall_ms_median": round(w[len(w) // 2], 2), "cpu_ms_median": round(c[len(c) // 2], 2)}
    out["total_wall_ms_median"] = round(sorted(sum(x[k][0] for k in x) for x in rows)[len(rows) // 2], 2)
    return out


for _ in range(3):
    frame()
static = [frame() for _ in range(cli.frames)]
moving = []
for t in range(cli.frames):
    env.take_action(base + 0.03 * np.sin(0.3 * t) * arm, action_type="qpos")
    moving.append(frame())
res = {"variant": VARIANT, "task": cli.task, "seed": cli.seed, "setup_s": round(setup_s, 2),
       "settings": {"shader": sapien.render.get_camera_shader_dir(),
                    "spp": sapien.render.get_ray_tracing_samples_per_pixel(),
                    "depth": sapien.render.get_ray_tracing_path_depth(),
                    "denoiser": sapien.render.get_ray_tracing_denoiser()},
       "static": summarize(static), "moving": summarize(moving), "cuda_sched": cuda_sched.STATUS,
       "frame_hashes": HASHES}
Path(cli.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res))
