"""Summarise scale_jobs.sh output: per K, wall of the batch, per-job rollout s/step, bitwise check vs K=1 j1."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = None
for kd in sorted(root.glob("k*"), key=lambda p: int(p.name[1:])):
    if not (kd / "t_end").exists():
        continue
    wall = float((kd / "t_end").read_text()) - float((kd / "t_start").read_text())
    rows, eq = [], []
    for jd in sorted(kd.glob("j*")):
        eps = [json.loads(l) for l in open(jd / "episodes.jsonl")] if (jd / "episodes.jsonl").exists() else []
        steps = sum(e["steps"] for e in eps)
        roll = sum(e["t_end"] - e["t_start"] for e in eps)
        fp = [(e["seed"], e["steps"], e["success"], tuple(e["consumed_obs_sha"]), tuple(e["chunk_sha"]), e["video_sha"]) for e in eps]
        if base is None:
            base = fp
        eq.append(fp == base)
        rows.append((len(eps), steps, round(roll / max(steps, 1) * 1000, 1)))
    tot_steps = sum(r[1] for r in rows)
    print(f"K={kd.name[1:]} batch_wall={wall:.1f}s jobs={rows} (eps, steps, ms/step rollout) "
          f"throughput={tot_steps / wall:.2f} steps/s  bitwise_vs_K1={eq}")
