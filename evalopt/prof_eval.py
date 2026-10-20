"""Run the official RoboTwin evaluator unchanged, with timing wrappers and per-episode fingerprints.

Nothing in the evaluation path is replaced: every wrapper calls the original function with the
original arguments and returns its result. Wrappers only read clocks and hash data that is
already being sent/returned, so they draw no random numbers and touch no simulator state.

Optimization candidates are opt-in via environment variables (all default off = official):
  EVALOPT_SKIP_UNUSED_CAMS=1  render only cameras the policy client reads (head/left/right)
  EVALOPT_LAZY_OBS=1          mid-chunk steps render only the head camera (eval video) and skip
                              update_obs; see opt_loop.py for the exact diff vs upstream
  EVALOPT_EXPERT_RECORD=f     store every seed's expert-check outcome (expert_cache.py)
  EVALOPT_EXPERT_REPLAY=f     skip the expert-check simulation, replay the stored outcomes
  EVALOPT_LAZY_CUROBO=1       build each cuRobo planner on first use instead of in set_planner
  EVALOPT_NO_VIDEO=1          (E3, protocol change) eval_video_log off: no eval video, no head frame per step
  EVALOPT_NO_SUBSTEP_RENDER=1 take_action without the per-physics-substep render sync (opt_take_action.py)
  EVALOPT_KEEP_CACHE=1        never call sapien's clear_cache (official clears every clear_cache_freq episodes)
  EVALOPT_FRAME_LOG=f         (with the camera patch) log every take_picture: camera, seed, ms
  EVALOPT_RECORD_ACTIONS=1    save every episode's raw action chunks to <out>/actions/<seed>.pkl
                              (replay_video.py rebuilds the eval video from them)
  EVALOPT_SERVER_NEW_JOB=1    at job start ask the (possibly reused) policy server to restart its
                              random-number chain at key(0) and drop its observation (prof_server.py)

Resident worker: --jobs JOBS.json runs several jobs in this one process (import, Sapien_TEST and
the renderer are paid once). JOBS.json = [{"out": dir, "args": [...], "record": f|null,
"replay": f|null}, ...]; each job writes the same files to its own "out".

usage (sim env active, cwd = RoboTwin root):
  python prof_eval.py --prof_out DIR -- <official eval_policy_xpolicylab.py args>
"""
import argparse
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_sched  # noqa: E402,F401  (EVALOPT_CUDA_BLOCKING_SYNC, before any CUDA init)
sys.path.insert(0, str(ROBOTWIN / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--prof_out")
pre.add_argument("--jobs")
pre_args, rest = pre.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]
EVAL_SCRIPT = str(ROBOTWIN / "scripts" / "eval_policy_xpolicylab.py")
sys.argv = [EVAL_SCRIPT] + rest
JOBS = json.load(open(pre_args.jobs)) if pre_args.jobs else None
OUT = Path(JOBS[0]["out"] if JOBS else pre_args.prof_out)
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.perf_counter()
WALL0 = time.time()

import numpy as np  # noqa: E402

import eval_policy_xpolicylab as E  # noqa: E402
import envs._base_task as BT  # noqa: E402
from envs.camera import camera as CAM  # noqa: E402
from envs.robot import robot as ROB  # noqa: E402
from envs.robot import planner as PLN  # noqa: E402
import sapien  # noqa: E402

T_IMPORT = time.perf_counter() - T0

# (ctx, name) -> [count, total_s, max_s]
STATS: dict = {}
CTX = ["top"]
CUR = {"seed": None, "task_env": None}
EP = {}
EVENTS = EPISODES = None
JOB_T0 = T0


def open_job(out):
    """Point all per-job outputs at `out` and reset the counters."""
    global OUT, EVENTS, EPISODES, JOB_T0
    for f in (EVENTS, EPISODES):
        if f is not None:
            f.close()
    OUT = Path(out)
    OUT.mkdir(parents=True, exist_ok=True)
    STATS.clear()
    EVENTS = open(OUT / "events.jsonl", "w")
    EPISODES = open(OUT / "episodes.jsonl", "w")
    JOB_T0 = time.perf_counter()


open_job(OUT)


def _add(name, dt):
    key = CTX[-1] + "|" + name
    s = STATS.setdefault(key, [0, 0.0, 0.0])
    s[0] += 1
    s[1] += dt
    if dt > s[2]:
        s[2] = dt


def timed(owner, attr, name=None, coarse=False, ctx=False):
    fn = getattr(owner, attr)
    if getattr(fn, "_prof_wrapped", False):
        return
    label = name or attr

    @functools.wraps(fn)
    def w(*a, **k):
        if coarse and "seed" in k:
            CUR["seed"] = k["seed"]
        t0 = time.perf_counter()
        if ctx:
            CTX.append(label)
        try:
            return fn(*a, **k)
        finally:
            if ctx:
                CTX.pop()
            dt = time.perf_counter() - t0
            _add(label, dt)
            if coarse:
                EVENTS.write(json.dumps({"n": label, "t": round(t0 - T0, 4), "d": round(dt, 4),
                                         "seed": CUR["seed"], "ctx": CTX[-1]}) + "\n")
                EVENTS.flush()

    w._prof_wrapped = True
    setattr(owner, attr, w)


# ---- simulator side ----
# EVALOPT_NO_SUBSTEP_RENDER=1: must be installed before the timers below wrap take_action
if os.environ.get("EVALOPT_NO_SUBSTEP_RENDER") == "1":
    import opt_take_action

    BT.Base_Task.take_action = opt_take_action.take_action_opt

for attr in ("_init_task_env_", "setup_scene", "create_table_and_wall", "load_robot", "load_camera",
             "check_stable", "get_cluttered_table", "close_env"):
    timed(BT.Base_Task, attr, coarse=True, ctx=True)
for attr in ("get_obs", "_update_render", "take_action", "take_dense_action", "move",
             "together_move_to_pose", "left_move_to_pose", "right_move_to_pose", "delay"):
    timed(BT.Base_Task, attr, ctx=attr in ("get_obs", "take_action"))
timed(CAM.Camera, "update_picture")
timed(CAM.Camera, "get_rgb")
timed(CAM.Camera, "get_config")
timed(sapien.Scene, "step", name="scene.step")
timed(sapien.Scene, "update_render", name="scene.update_render")
timed(ROB.Robot, "reset", name="robot.reset")
timed(ROB.Robot, "set_planner", name="robot.set_planner", coarse=True)
for attr in ("left_plan_path", "right_plan_path", "left_plan_multi_path", "right_plan_multi_path",
             "set_arm_joints", "set_gripper", "get_left_arm_jointState", "get_right_arm_jointState"):
    timed(ROB.Robot, attr, name="robot." + attr)
timed(CAM.Camera, "update_wrist_camera")
import mplib  # noqa: E402

timed(mplib.Planner, "TOPP", name="mplib.TOPP")
_orig_clear = BT.sapien_clear_cache


def _clear():
    if os.environ.get("EVALOPT_KEEP_CACHE") == "1":
        _add("sapien_clear_cache_skipped", 0.0)
        return
    t0 = time.perf_counter()
    _orig_clear()
    _add("sapien_clear_cache", time.perf_counter() - t0)


BT.sapien_clear_cache = _clear

import expert_cache  # noqa: E402

EC = expert_cache.ExpertCache()
_orig_class_decorator = E.class_decorator


def class_decorator(task_name):
    t0 = time.perf_counter()
    env = _orig_class_decorator(task_name)
    _add("class_decorator", time.perf_counter() - t0)
    cls = type(env)
    if JOBS or EC.active():
        expert_cache.ExpertCache.install(cls)
    timed(cls, "setup_demo", coarse=True, ctx=True)
    timed(cls, "load_actors", ctx=True)
    timed(cls, "play_once", coarse=True, ctx=True)
    timed(cls, "check_success")
    CUR["task_env"] = env
    return env


E.class_decorator = class_decorator
timed(E, "build_instruction", coarse=True)
timed(E, "robotwin_obs_to_xpolicylab", name="obs_to_xpl")

# ---- optimization candidates (opt-in) ----
SKIP_UNUSED = os.environ.get("EVALOPT_SKIP_UNUSED_CAMS") == "1"
LAZY_OBS = os.environ.get("EVALOPT_LAZY_OBS") == "1"
import opt_loop  # noqa: E402


def _wanted(cams, name):
    only = getattr(cams, "_evalopt_render_only", None)
    if only is None and SKIP_UNUSED:
        only = opt_loop.POLICY_CAMS
    return only is None or name in only


def _get_rgba_img(camera):
    camera_rgba = camera.get_picture("Color")
    return (camera_rgba * 255).clip(0, 255).astype("uint8")


_FRAME_LOG = os.environ.get("EVALOPT_FRAME_LOG")
_frame_log = open(_FRAME_LOG, "a", buffering=1) if _FRAME_LOG else None


def _logged_take(cam, name):
    if _frame_log is None:
        return cam.take_picture()
    t = time.perf_counter()
    cam.take_picture()
    _frame_log.write(json.dumps({"t": round(time.perf_counter() - T0, 4), "cam": name, "seed": CUR["seed"],
                                 "ms": round((time.perf_counter() - t) * 1e3, 2)}) + "\n")


def update_picture(self):
    if self.collect_wrist_camera:
        if _wanted(self, "left_camera"):
            _logged_take(self.left_camera, "left_camera")
        if _wanted(self, "right_camera"):
            _logged_take(self.right_camera, "right_camera")
    for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
        if _wanted(self, camera_name):
            _logged_take(camera, camera_name)


def get_rgba(self):
    res = {}
    if self.collect_wrist_camera:
        for name, cam in (("left_camera", self.left_camera), ("right_camera", self.right_camera)):
            if _wanted(self, name):
                res[name] = {"rgba": _get_rgba_img(cam)}
    for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
        if camera_name == "head_camera" and not self.collect_head_camera:
            continue
        if _wanted(self, camera_name):
            res[camera_name] = {"rgba": _get_rgba_img(camera)}
    return res


if SKIP_UNUSED or LAZY_OBS:
    CAM.Camera.update_picture = update_picture
    CAM.Camera.get_rgba = get_rgba
    timed(CAM.Camera, "update_picture")
if LAZY_OBS:
    E.eval_remote_policy = opt_loop.eval_remote_policy_opt


# eval video fingerprint: hash every frame written to the official ffmpeg pipe
class _HashPipe:
    def __init__(self, pipe):
        self._pipe = pipe

    def write(self, b):
        if EP:
            EP["h_video"].update(b)
            EP["n_frames"] += 1
            if b == EP.get("_last_frame"):
                EP["n_frames_repeat"] = EP.get("n_frames_repeat", 0) + 1
            EP["_last_frame"] = b
        t0 = time.perf_counter()
        try:
            return self._pipe.write(b)
        finally:
            _add("video.pipe_write", time.perf_counter() - t0)

    def __getattr__(self, k):
        return getattr(self._pipe, k)


_orig_set_ffmpeg = BT.Base_Task._set_eval_video_ffmpeg


def _set_eval_video_ffmpeg(self, ffmpeg):
    ffmpeg.stdin = _HashPipe(ffmpeg.stdin)
    return _orig_set_ffmpeg(self, ffmpeg)


BT.Base_Task._set_eval_video_ffmpeg = _set_eval_video_ffmpeg

# EVALOPT_DUMMY_POLICY=1: no policy server; every episode ends after one no-op action (used only to
# test that expert-check outcomes do not depend on what the policy did in earlier episodes)
class _DummyClient:
    _robotwin_protocol = "ws"

    def call(self, func_name=None, obs=None, **_):
        env = CUR["task_env"]
        if func_name == "get_action":
            env.take_action_cnt = env.step_lim
            return [np.asarray(env.now_obs["joint_action"]["vector"], dtype=np.float32)]
        return None

    def close(self):
        pass


if os.environ.get("EVALOPT_DUMMY_POLICY") == "1":
    E.build_policy_client = lambda usr_args: _DummyClient()

# EVALOPT_LAZY_CUROBO=1: cuRobo planners are built (MotionGen + warmup, ~24 s) on first use; with
# replayed expert checks and joint-space actions the eval phase never uses them.
_RealCuroboPlanner = ROB.CuroboPlanner


class LazyCuroboPlanner:
    def __init__(self, *a, **k):
        object.__setattr__(self, "_lazy_args", (a, k))
        object.__setattr__(self, "_lazy_obj", None)

    def _lazy_real(self):
        obj = object.__getattribute__(self, "_lazy_obj")
        if obj is None:
            a, k = object.__getattribute__(self, "_lazy_args")
            t0 = time.perf_counter()
            obj = _RealCuroboPlanner(*a, **k)
            _add("curobo.lazy_build", time.perf_counter() - t0)
            object.__setattr__(self, "_lazy_obj", obj)
        return obj

    def plan_grippers(self, now_val, target_val):
        # Upstream CuroboPlanner.plan_grippers is a pure function of its arguments (linspace over
        # 200 steps, never touches self); calling it on the proxy is the same code and the same
        # numbers, and does not force a MotionGen build (_init_task_env_ opens the grippers with it).
        _add("curobo.plan_grippers_unbuilt", 0.0)
        return _RealCuroboPlanner.plan_grippers(self, now_val, target_val)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_lazy_real")(), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_lazy_real")(), name, value)


if os.environ.get("EVALOPT_LAZY_CUROBO") == "1":
    ROB.CuroboPlanner = LazyCuroboPlanner

# EVALOPT_SERVER_NEW_JOB=1: a reused policy server is put back into the state of a fresh one
_orig_build_client = E.build_policy_client


def build_policy_client(usr_args):
    client = _orig_build_client(usr_args)
    if os.environ.get("EVALOPT_SERVER_NEW_JOB") == "1":
        t0 = time.perf_counter()
        client.call(func_name="evalopt_new_job")
        _add("client.evalopt_new_job", time.perf_counter() - t0)
    return client


E.build_policy_client = build_policy_client

# EVALOPT_NO_VIDEO=1: turn the task config's eval_video_log off (protocol change, see GOAL.md E3)
_orig_load_task_args = E.load_task_args


def load_task_args(usr_args):
    args, name = _orig_load_task_args(usr_args)
    if os.environ.get("EVALOPT_NO_VIDEO") == "1":
        args["eval_video_log"] = False
    return args, name


E.load_task_args = load_task_args
RECORD_ACTIONS = os.environ.get("EVALOPT_RECORD_ACTIONS") == "1"

# ---- policy client side ----
from client_server.ws import WsModelClient  # noqa: E402
from client_server.ws.protocol import codec as CODEC  # noqa: E402


def _hash_obs(h, obs):
    for cam in sorted(obs.get("vision", {})):
        c = obs["vision"][cam].get("color")
        if c is not None:
            h.update(cam.encode())
            h.update(np.ascontiguousarray(c).tobytes())
    st = obs.get("state", {})
    for k in sorted(st):
        h.update(k.encode())
        h.update(np.ascontiguousarray(st[k]).tobytes())
    h.update(str(obs.get("instruction")).encode())


def _hash_action(h, act):
    if isinstance(act, dict):
        for k in sorted(act):
            h.update(k.encode())
            _hash_action(h, act[k])
    elif isinstance(act, (list, tuple)):
        for a in act:
            _hash_action(h, a)
    else:
        h.update(np.ascontiguousarray(np.asarray(act)).tobytes())


_orig_call = WsModelClient.call


def call(self, *a, **k):
    fname = k.get("func_name") or (a[0] if a else "?")
    obs = k.get("obs")
    if fname == "prepare_case" and isinstance(obs, dict):
        EP.clear()
        EP.update(seed=obs.get("seed"), instruction=obs.get("instruction"), n_obs=0, n_act=0,
                  h_obs=hashlib.sha256(), h_act=hashlib.sha256(), h_first_act=None,
                  t_start=time.perf_counter() - T0, infer_rtt=[], last_obs=None,
                  consumed_obs=[], chunk_sha=[], h_video=hashlib.sha256(), n_frames=0, chunks=[])
    t0 = time.perf_counter()
    res = _orig_call(self, *a, **k)
    dt = time.perf_counter() - t0
    _add("client." + str(fname), dt)
    if EP:
        if fname == "update_obs":
            EP["n_obs"] += 1
            _hash_obs(EP["h_obs"], obs)
            ho = hashlib.sha256()
            _hash_obs(ho, obs)
            EP["last_obs"] = ho.hexdigest()[:12]
        elif fname == "get_action":
            EP["n_act"] += 1
            EP["infer_rtt"].append(round(dt * 1000, 2))
            hh = hashlib.sha256()
            _hash_action(hh, res)
            if EP["h_first_act"] is None:
                EP["h_first_act"] = hh.hexdigest()[:16]
            EP["h_act"].update(hh.digest())
            EP["consumed_obs"].append(EP["last_obs"])
            EP["chunk_sha"].append(hh.hexdigest()[:12])
            if RECORD_ACTIONS:
                EP["chunks"].append(res)
        elif fname == "trial_end" and isinstance(obs, dict):
            env = CUR["task_env"]
            rec = {"seed": EP.get("seed"), "instruction": EP.get("instruction"),
                   "success": obs.get("success"), "steps": getattr(env, "take_action_cnt", None),
                   "n_update_obs": EP["n_obs"], "n_get_action": EP["n_act"],
                   "obs_sha": EP["h_obs"].hexdigest()[:24], "act_sha": EP["h_act"].hexdigest()[:24],
                   "first_act_sha": EP["h_first_act"], "t_start": round(EP["t_start"], 3),
                   "t_end": round(time.perf_counter() - T0, 3), "infer_rtt_ms": EP["infer_rtt"],
                   "consumed_obs_sha": EP["consumed_obs"], "chunk_sha": EP["chunk_sha"],
                   "video_sha": EP["h_video"].hexdigest()[:24], "video_frames": EP["n_frames"],
                   "video_frames_repeat": EP.get("n_frames_repeat", 0),
                   "opt": {"skip_unused_cams": SKIP_UNUSED, "lazy_obs": LAZY_OBS,
                           "expert_replay": bool(expert_cache.REPLAY)}}
            if RECORD_ACTIONS:
                import pickle

                (OUT / "actions").mkdir(exist_ok=True)
                with open(OUT / "actions" / f"{rec['seed']}.pkl", "wb") as f:
                    pickle.dump({"seed": rec["seed"], "chunks": EP["chunks"], "steps": rec["steps"],
                                 "success": rec["success"]}, f)
            EPISODES.write(json.dumps(rec) + "\n")
            EPISODES.flush()
            EP.clear()
    return res


WsModelClient.call = call
for attr in ("encode_frame", "decode_frame"):
    if hasattr(CODEC, attr):
        timed(CODEC, attr, name="codec." + attr)


def dump():
    EC.save({"argv": sys.argv})
    total = time.perf_counter() - JOB_T0
    rows = sorted(({"key": k, "count": v[0], "total_s": round(v[1], 4), "max_s": round(v[2], 4)}
                   for k, v in STATS.items()), key=lambda r: -r["total_s"])
    with open(OUT / "stats.json", "w") as f:
        json.dump({"wall_s": round(total, 3), "import_s": round(T_IMPORT, 3) if OUT_FIRST[0] == OUT else 0.0,
                   "argv": sys.argv, "wall_start": WALL0, "resident_worker": bool(JOBS),
                   "env": {k: os.environ.get(k) for k in
                           ("CUDA_VISIBLE_DEVICES", "VK_ICD_FILENAMES")},
                   "rows": rows}, f, indent=1)
    EVENTS.flush()
    EPISODES.flush()


OUT_FIRST = [OUT]


if __name__ == "__main__":
    from test_render import Sapien_TEST

    t0 = time.perf_counter()
    Sapien_TEST()
    _add("Sapien_TEST", time.perf_counter() - t0)
    if not JOBS:
        try:
            parsed = E.parse_args()
            E.main(parsed)
        finally:
            dump()
            EVENTS.close()
            EPISODES.close()
    else:
        for i, job in enumerate(JOBS):
            if i:
                open_job(job["out"])
            EC = expert_cache.ExpertCache(record=job.get("record") or "", replay=job.get("replay") or "")
            sys.argv = [EVAL_SCRIPT] + list(job["args"])
            try:
                E.main(E.parse_args())
                (OUT / "exit_code").write_text("0\n")
            except BaseException:
                import traceback

                (OUT / "exit_code").write_text("1\n")
                (OUT / "traceback.txt").write_text(traceback.format_exc())
            finally:
                dump()
        EVENTS.close()
        EPISODES.close()
