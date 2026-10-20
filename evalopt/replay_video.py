"""Rebuild the eval video of recorded episodes by replaying their action chunks (no policy).

For each <actions_dir>/<seed>.pkl (written by prof_eval.py EVALOPT_RECORD_ACTIONS=1) this rebuilds
the eval scene with the evaluator's own setup_demo(seed) and replays the chunks following the
official rollout loop's control flow (break on success / step limit / end of chunk). Every frame the
official take_action would pipe to ffmpeg is hashed the same way prof_eval.py hashes the real pipe,
so the result can be compared with a reference arm's video_sha / video_frames.
With --mp4_dir the frames also go through the evaluator's exact ffmpeg command, so the resulting
episode<k>.mp4 can be compared byte for byte with the official eval video (k = order in the ref arm).
Only the head camera is rendered (same Camera patch as prof_eval.py's EVALOPT_LAZY_OBS path).
usage: python replay_video.py --task T --config C --actions DIR --ref REF_DIR --out f.json [--mp4_dir D]
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROBOTWIN / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_policy_xpolicylab as E  # noqa: E402
import opt_loop  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--actions", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mp4_dir")
cli = ap.parse_args()

import subprocess  # noqa: E402

from envs.camera import camera as CAM  # noqa: E402


def _wanted(cams, name):
    only = getattr(cams, "_evalopt_render_only", None)
    return only is None or name in only


def update_picture(self):
    if self.collect_wrist_camera:
        if _wanted(self, "left_camera"):
            self.left_camera.take_picture()
        if _wanted(self, "right_camera"):
            self.right_camera.take_picture()
    for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
        if _wanted(self, camera_name):
            camera.take_picture()


def get_rgba(self):
    res = {}
    if self.collect_wrist_camera:
        for name, cam in (("left_camera", self.left_camera), ("right_camera", self.right_camera)):
            if _wanted(self, name):
                res[name] = {"rgba": (cam.get_picture("Color") * 255).clip(0, 255).astype("uint8")}
    for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
        if camera_name == "head_camera" and not self.collect_head_camera:
            continue
        if _wanted(self, camera_name):
            res[camera_name] = {"rgba": (camera.get_picture("Color") * 255).clip(0, 255).astype("uint8")}
    return res


CAM.Camera.update_picture = update_picture
CAM.Camera.get_rgba = get_rgba


class HashPipe:
    def __init__(self, real=None):
        self.h = hashlib.sha256()
        self.n = 0
        self.real = real

    def write(self, b):
        self.h.update(b)
        self.n += 1
        if self.real is not None:
            self.real.stdin.write(b)

    def close(self):
        pass


class FakeFfmpeg:
    def __init__(self, real=None):
        self.stdin = HashPipe(real)
        self.real = real

    def wait(self):
        if self.real is not None:
            self.real.stdin.close()
            self.real.wait()


usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "replay"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
args["eval_video_save_dir"] = "/nonexistent-replay"  # only needs to be non-None
env = E.class_decorator(cli.task)
ref = {e["seed"]: e for e in (json.loads(l) for l in open(Path(cli.ref) / "episodes.jsonl"))}
out = []
for p in sorted(Path(cli.actions).glob("*.pkl")):
    rec = pickle.load(open(p, "rb"))
    seed = rec["seed"]
    t0 = time.perf_counter()
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    real = None
    if cli.mp4_dir:
        # same command as eval_policy_xpolicylab.eval_remote_policy
        Path(cli.mp4_dir).mkdir(parents=True, exist_ok=True)
        cam = E.get_camera_config(args["camera"]["head_camera_type"])
        real = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                                 "-video_size", f"{cam['w']}x{cam['h']}", "-framerate", "10", "-i", "-",
                                 "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                                 f"{cli.mp4_dir}/episode{len(out)}.mp4"], stdin=subprocess.PIPE)
    ff = FakeFfmpeg(real)
    env._set_eval_video_ffmpeg(ff)
    for chunk in rec["chunks"]:
        if E.is_episode_end(env):
            break
        observation = opt_loop.get_obs_cams(env, opt_loop.VIDEO_CAMS)
        actions = E.normalize_action_chunk(chunk)
        for idx, action in enumerate(actions):
            flat, kind = E.xpolicylab_action_to_robotwin(action, action_type="joint", current_observation=observation)
            env.take_action(flat, action_type=kind)
            if env.eval_success or E.is_episode_end(env) or idx + 1 == len(actions):
                break
            observation = opt_loop.get_obs_cams(env, opt_loop.VIDEO_CAMS)
        if env.eval_success:
            break
    ff.wait()
    r = {"seed": seed, "steps": env.take_action_cnt, "success": bool(env.eval_success),
         "video_sha": ff.stdin.h.hexdigest()[:24], "video_frames": ff.stdin.n,
         "replay_s": round(time.perf_counter() - t0, 1)}
    rf = ref.get(seed, {})
    r["matches_ref"] = {k: r[k] == rf.get(k) for k in ("steps", "success", "video_sha", "video_frames")}
    env.close_env()
    out.append(r)
    print(json.dumps(r), flush=True)
Path(cli.out).write_text(json.dumps(out, indent=1))
