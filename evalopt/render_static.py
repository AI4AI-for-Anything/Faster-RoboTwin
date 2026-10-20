"""Does a camera's image depend on whether it was rendered at earlier steps (RT accumulation)?

T1  setup -> get_obs x3 with no physics in between: are repeated renders of an unchanged
    scene identical?
T2  setup -> hold the arms at their current joints (the action is the current qpos) for N steps:
      official : get_obs() after every action (all cameras)
      skip     : no camera rendering until the last step (_update_render still called)
    compare the last-step image of each camera, and whether the scene was static at the end.
usage: python render_static.py --out f.json [--task adjust_bottle --seed 100001 --steps 25]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROBOTWIN = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROBOTWIN / "scripts"))

import numpy as np  # noqa: E402

import eval_policy_xpolicylab as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="adjust_bottle")
ap.add_argument("--config", default="demo_clean")
ap.add_argument("--seed", type=int, default=100001)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--out", required=True)
cli = ap.parse_args()

usr = {"task_name": cli.task, "task_config": cli.config, "policy_name": "Pi_05", "ckpt_setting": "bench"}
args, _ = E.load_task_args(usr)
args["eval_mode"] = True
env = E.class_decorator(cli.task)


def setup():
    env.setup_demo(now_ep_num=0, seed=cli.seed, is_test=True, **args)


def imgs(obs):
    return {k: hashlib.sha256(np.ascontiguousarray(v["rgb"]).tobytes()).hexdigest()[:12]
            for k, v in obs["observation"].items() if "rgb" in v}


def poses():
    m = hashlib.sha256()
    for a in env.scene.get_all_actors():
        p = a.get_pose()
        m.update(np.asarray(p.p, dtype=np.float32).tobytes())
        m.update(np.asarray(p.q, dtype=np.float32).tobytes())
    for e in (env.robot.left_entity, env.robot.right_entity):
        m.update(np.asarray(e.get_qpos(), dtype=np.float32).tobytes())
    return m.hexdigest()[:12]


res = {}
setup()
res["T1"] = [imgs(env.get_obs()) for _ in range(3)]
env.close_env()


def hold(schedule):
    setup()
    obs = env.get_obs()
    hold_a = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    per_step, pose_seq = [], []
    for t in range(cli.steps):
        env.take_action(hold_a, action_type="qpos")
        pose_seq.append(poses())
        if schedule == "official" or t + 1 == cli.steps:
            per_step.append(imgs(env.get_obs()))
        else:
            env._update_render()
    env.close_env()
    return {"last": per_step[-1], "per_step": per_step, "poses": pose_seq}


res["official"] = hold("official")
res["skip"] = hold("skip")
o, s = res["official"], res["skip"]
res["summary"] = {
    "T1_repeat_identical": [res["T1"][0] == x for x in res["T1"][1:]],
    "last_images_equal": {k: o["last"][k] == s["last"][k] for k in o["last"]},
    "poses_equal_seq": o["poses"] == s["poses"],
    "static_tail_steps": sum(1 for i in range(1, len(o["poses"])) if o["poses"][i] == o["poses"][i - 1]),
    "official_consecutive_identical_images": {
        k: sum(1 for i in range(1, len(o["per_step"])) if o["per_step"][i][k] == o["per_step"][i - 1][k])
        for k in o["last"]},
}
Path(cli.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res["summary"], indent=1))
