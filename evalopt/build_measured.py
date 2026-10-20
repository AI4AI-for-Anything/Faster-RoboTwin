"""Build measured.json for full_estimate.py from official reference arms and our saturated runs.

Official s_off excludes each arm's first episode (its first-frame pipeline compile and first
inference are one-off per job and are counted in full_estimate.J_OFF instead). clear_included is set
when the arm ran >= 6 episodes (sapien clear_cache every 5 episodes happened inside the rollout).
e_off needs an arm that ran the expert check for real (not replayed).
usage: python build_measured.py spec.json out.json
  spec.json: {"task:setting": {"rollout_ref": dir, "expert_ref": dir, "ours": run_dir}, ...}
"""
import json
import os
import subprocess
import sys

spec = json.load(open(sys.argv[1]))
here = os.path.dirname(os.path.abspath(__file__))
out = {}
for key, sp in spec.items():
    eps = [json.loads(l) for l in open(os.path.join(sp["rollout_ref"], "episodes.jsonl"))]
    rest = eps[1:]
    s_off = sum(e["t_end"] - e["t_start"] for e in rest) / sum(e["steps"] for e in rest)
    ev = [json.loads(l) for l in open(os.path.join(sp["expert_ref"], "events.jsonl"))]
    eps_x = [json.loads(l) for l in open(os.path.join(sp["expert_ref"], "episodes.jsonl"))]
    sd = [x for x in ev if x["n"] == "setup_demo" and x["ctx"] == "top"]
    po = [x for x in ev if x["n"] == "play_once" and x["ctx"] == "top"]
    e_off = (sum(x["d"] for x in sd[1:]) + sum(x["d"] for x in po)) / len(eps_x)
    rep = json.loads(subprocess.check_output([sys.executable, os.path.join(here, "throughput_report.py"), sp["ours"]]))
    out[key] = {"s_off": round(s_off, 4), "e_off": round(e_off, 2), "thr": rep["steady_steps_per_s"],
                "clear_included": len(eps) >= 6, "official_eps_used": len(rest), "ours_run": rep["run"],
                "ours_workers": rep["workers"]}
    print(key, out[key])
json.dump(out, open(sys.argv[2], "w"), indent=1)
