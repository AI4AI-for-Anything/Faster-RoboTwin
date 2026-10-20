"""Bitwise per-episode comparison of two prof_eval.py output dirs (same fields as compare_arms.py).
usage: python compare_dirs.py <ref_dir> <cand_dir>"""
import json
import sys

KEYS = ("seed", "instruction", "success", "steps", "n_get_action", "consumed_obs_sha", "chunk_sha",
        "video_sha", "video_frames")
a, b = ([json.loads(l) for l in open(d + "/episodes.jsonl")] for d in sys.argv[1:3])
res = [[k for k in KEYS if x.get(k) != y.get(k)] for x, y in zip(a, b)]
ok = len(a) == len(b) and not any(res)
print(json.dumps({"ref": sys.argv[1].split("/runs/")[-1], "cand": sys.argv[2].split("/runs/")[-1],
                  "episodes": [len(a), len(b)], "bitwise_equal": ok,
                  "mismatch_fields": [r for r in res if r]}))
