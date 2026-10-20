"""Per-GPU throughput of a run_ms_workers.sh / run_workers.sh output dir, plus bitwise checks.

Steady-state window = from the moment the last worker finished its first job's start-up until the
first worker finished its whole job list (all K workers busy). Throughput there = steps completed
inside the window / window length, using each episode's [t_start, t_end] (pro-rated).
usage: python throughput_report.py <run_dir> [<ref_dir_for_bitwise> ...]
"""
import glob
import json
import os
import sys

KEYS = ("seed", "instruction", "success", "steps", "n_get_action", "consumed_obs_sha", "chunk_sha",
        "video_sha", "video_frames")
run = sys.argv[1]
refs = {}
for r in sys.argv[2:]:
    eps = [json.loads(l) for l in open(os.path.join(r, "episodes.jsonl"))]
    refs[eps[0]["instruction"].split()[0] + str(len(eps)) + str(eps[0]["seed"])] = eps
t_start = float(open(os.path.join(run, "t_start")).read())
workers = sorted(glob.glob(os.path.join(run, "w*/")))
spans, eps_all, bit = [], [], []
for w in workers:
    jobs = sorted(glob.glob(os.path.join(w, "j*/")))
    # episode times are relative to prof_eval.py's T0, taken together with WALL0 (stats.json wall_start)
    sj = [os.path.join(j, "stats.json") for j in jobs if os.path.exists(os.path.join(j, "stats.json"))]
    wt0 = json.load(open(sj[0]))["wall_start"] if sj else float(open(os.path.join(w, "t_start")).read())
    first_ep, last_ep = None, None
    for j in jobs:
        st = json.load(open(os.path.join(j, "stats.json"))) if os.path.exists(os.path.join(j, "stats.json")) else None
        eps = [json.loads(l) for l in open(os.path.join(j, "episodes.jsonl"))] if os.path.exists(os.path.join(j, "episodes.jsonl")) else []
        for e in eps:
            a, b = wt0 + e["t_start"], wt0 + e["t_end"]
            eps_all.append((a, b, e["steps"]))
            first_ep = a if first_ep is None else min(first_ep, a)
            last_ep = b if last_ep is None else max(last_ep, b)
        for key, ref in refs.items():
            if eps and len(eps) == len(ref) and eps[0]["seed"] == ref[0]["seed"] and eps[0]["instruction"] == ref[0]["instruction"]:
                bit.append(all(x.get(k) == y.get(k) for x, y in zip(eps, ref) for k in KEYS))
    spans.append((first_ep, last_ep))
w0 = max(s[0] for s in spans)
w1 = min(s[1] for s in spans)
steps_in = 0.0
for a, b, n in eps_all:
    lo, hi = max(a, w0), min(b, w1)
    if hi > lo and b > a:
        steps_in += n * (hi - lo) / (b - a)
total_steps = sum(n for _, _, n in eps_all)
t_end = float(open(os.path.join(run, "t_end")).read()) if os.path.exists(os.path.join(run, "t_end")) else max(s[1] for s in spans)
print(json.dumps({"run": run.rstrip("/").split("/runs/")[-1], "workers": len(workers), "jobs": sum(len(glob.glob(os.path.join(w, 'j*/'))) for w in workers),
                  "steady_window_s": round(w1 - w0, 1), "steady_steps_per_s": round(steps_in / max(w1 - w0, 1e-9), 2),
                  "batch_wall_s": round(t_end - t_start, 1), "batch_steps_per_s": round(total_steps / (t_end - t_start), 2),
                  "total_steps": total_steps, "bitwise_checked": len(bit), "bitwise_all_equal": all(bit) if bit else None}))
