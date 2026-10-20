"""Per-step / per-episode / per-job cost table from one prof_eval.py output directory.
usage: python stats_report.py <arm_dir> [<arm_dir> ...]
"""
import json
import sys
from pathlib import Path

for d in map(Path, sys.argv[1:]):
    s = json.load(open(d / "stats.json"))
    rows = {r["key"]: r for r in s["rows"]}
    eps = [json.loads(l) for l in open(d / "episodes.jsonl")]
    steps = sum(e["steps"] for e in eps)
    roll = sum(e["t_end"] - e["t_start"] for e in eps)

    def tot(*keys):
        return sum(rows[k]["total_s"] for k in keys if k in rows)

    def cnt(k):
        return rows[k]["count"] if k in rows else 0

    srv = [json.loads(l) for l in open(d / "server_prof.jsonl")] if (d / "server_prof.jsonl").exists() else []
    infer = [x["d"] for x in srv if x["n"] == "policy_infer"]
    items = {
        "get_obs (all)": tot("top|get_obs"),
        "  take_picture (update_picture)": tot("get_obs|update_picture"),
        "  get_picture+convert (get_rgb)": tot("get_obs|get_rgb"),
        "take_action (all)": tot("top|take_action"),
        "  scene.step": tot("take_action|scene.step"),
        "  mplib.TOPP": tot("take_action|mplib.TOPP"),
        "  _update_render (per substep)": tot("take_action|_update_render"),
        "  check_success (per substep)": tot("take_action|check_success"),
        "  robot.set_arm_joints": tot("take_action|robot.set_arm_joints"),
        "  robot.set_gripper": tot("take_action|robot.set_gripper"),
        "  joint-state reads": tot("take_action|robot.get_left_arm_jointState", "take_action|robot.get_right_arm_jointState"),
        "  video pipe write": tot("take_action|video.pipe_write"),
        "client.get_action (infer RTT)": tot("top|client.get_action"),
        "client.update_obs": tot("top|client.update_obs"),
        "obs_to_xpl": tot("top|obs_to_xpl"),
    }
    print(f"== {d.name}: wall {s['wall_s']:.1f}s, import {s['import_s']:.1f}s, episodes {len(eps)}, steps {steps}, "
          f"rollout {roll:.1f}s ({roll / max(steps, 1) * 1000:.0f} ms/step)")
    for k, v in items.items():
        print(f"   {k:36s} {v:8.2f}s  {v / max(steps, 1) * 1000:7.1f} ms/step  {v / max(roll, 1e-9) * 100:5.1f}% of rollout")
    print(f"   {'take_action count':36s} {cnt('top|take_action')}  scene.step count {cnt('take_action|scene.step')}")
    fixed = {
        "Sapien_TEST": tot("top|Sapien_TEST"),
        "set_planner (cuRobo build+warmup)": tot("load_robot|robot.set_planner"),
        "setup_demo (all)": tot("top|setup_demo"),
        "play_once (expert)": tot("top|play_once"),
        "close_env": tot("top|close_env"),
        "build_instruction": tot("top|build_instruction"),
    }
    print("   fixed:", ", ".join(f"{k} {v:.1f}s" for k, v in fixed.items()),
          f"| first infer {infer[0]:.1f}s, steady infer median {sorted(infer[1:])[len(infer[1:]) // 2] * 1000:.0f} ms ({len(infer)} calls)" if len(infer) > 1 else "")
    other = s["wall_s"] - s["import_s"] - roll - fixed["Sapien_TEST"] - fixed["setup_demo (all)"] - fixed["play_once (expert)"]
    print(f"   unaccounted (wall - import - rollout - Sapien_TEST - setup - expert): {other:.1f}s")
