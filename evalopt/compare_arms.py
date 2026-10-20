"""Per-episode bitwise judge: compare two profiled arms (episodes.jsonl from prof_eval.py).

An episode matches when every recorded fingerprint is equal: seed, instruction, success, steps,
number of policy queries, the hash of every observation the policy consumed, the hash of every
action chunk it returned, and the hash of every eval-video frame. Also prints wall-clock per arm.
usage: python compare_arms.py <case_dir> [armA armB ...]   (default: ref vs every other arm)
"""
import json
import sys
from pathlib import Path

KEYS = ("seed", "instruction", "success", "steps", "n_get_action", "consumed_obs_sha", "chunk_sha",
        "video_sha", "video_frames")


def load(d):
    p = d / "episodes.jsonl"
    eps = [json.loads(l) for l in open(p)] if p.exists() else []
    st = json.load(open(d / "stats.json")) if (d / "stats.json").exists() else {}
    ec = (d / "exit_code").read_text().strip() if (d / "exit_code").exists() else "?"
    return eps, st, ec


case = Path(sys.argv[1])
arms = sys.argv[2:] or sorted(p.name for p in case.iterdir() if p.is_dir())
ref_name = "ref" if "ref" in arms else arms[0]
data = {a: load(case / a) for a in arms}
ref = data[ref_name][0]
out = {"case": case.name, "ref": ref_name, "arms": {}}
for a in arms:
    eps, st, ec = data[a]
    steps = sum(e["steps"] or 0 for e in eps)
    row = {"exit": ec, "episodes": len(eps), "success": sum(bool(e["success"]) for e in eps),
           "steps": steps, "wall_s": st.get("wall_s"),
           "rollout_s": round(sum(e["t_end"] - e["t_start"] for e in eps), 1)}
    if a != ref_name:
        mism = []
        for i, (r, c) in enumerate(zip(ref, eps)):
            bad = [k for k in KEYS if r.get(k) != c.get(k)]
            if bad:
                mism.append({"episode": i, "seed": r.get("seed"), "fields": bad})
        if len(ref) != len(eps):
            mism.append({"episode_count": [len(ref), len(eps)]})
        row["bitwise_equal"] = not mism
        row["mismatch"] = mism
        if data[ref_name][1].get("wall_s") and st.get("wall_s"):
            row["speedup_wall"] = round(data[ref_name][1]["wall_s"] / st["wall_s"], 3)
        rr = sum(e["t_end"] - e["t_start"] for e in ref)
        if rr and row["rollout_s"]:
            row["speedup_rollout"] = round(rr / row["rollout_s"], 3)
    out["arms"][a] = row
print(json.dumps(out))
