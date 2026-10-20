#!/usr/bin/env bash
# Start K render processes on one GPU, each probe_multienv.py with N scenes, wait until all are ready,
# release them together for <seconds>. Total scenes = K*N.
# usage: probe_scale.sh <gpu> <K> <N> <task> <config> <out_dir> [seconds]
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
GPU=$1; K=$2; N=$3; TASK=$4; CFG=$5; OUT=$6; SEC=${7:-120}
mkdir -p "$OUT"; rm -f "$OUT"/ready.* "$OUT"/go
source "$SIM_ACTIVATE"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONWARNINGS=ignore EVALOPT_CUDA_BLOCKING_SYNC=1
nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader -lms 1000 > "$OUT/gpu.csv" &
SMI=$!
for i in $(seq 1 "$K"); do
  timeout $((SEC + 900)) python "$EVALOPT_PY/probe_multienv.py" --n "$N" --task "$TASK" --config "$CFG" \
    --seconds "$SEC" --ready "$OUT/ready.$i" --go "$OUT/go" --out "$OUT/p$i.jsonl" > "$OUT/p$i.log" 2>&1 &
done
for t in $(seq 1 900); do [ "$(ls "$OUT"/ready.* 2>/dev/null | wc -l)" -ge "$K" ] && break; sleep 1; done
date +%s.%N > "$OUT/t_go"; touch "$OUT/go"
wait $(jobs -p | grep -v "^$SMI$") 2>/dev/null
kill $SMI 2>/dev/null
python3 - "$OUT" "$K" "$N" <<'PY'
import glob, json, sys, statistics as st
out, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
runs = []
for f in sorted(glob.glob(out + "/p*.jsonl")):
    L = [json.loads(l) for l in open(f)]
    runs.append([x for x in L if "t" in x])
runs = [r for r in runs if r]
t0 = max(r[0]["t"] for r in runs) + 10; t1 = min(r[-1]["t"] for r in runs) - 5
n = sum(1 for r in runs for x in r if t0 <= x["t"] <= t1)
ms = [x["ms"] for r in runs for x in r if t0 <= x["t"] <= t1]
u = []
for line in open(out + "/gpu.csv"):
    try: u.append(float(line.split(",")[1].split()[0]))
    except Exception: pass
res = {"K_procs": K, "N_scenes_per_proc": N, "procs_ok": len(runs), "window_s": round(t1 - t0, 1),
       "frames_per_s": round(n / (t1 - t0), 2), "frame_ms_median": round(st.median(ms), 1),
       "gpu_util_mean": round(sum(u[len(u)//4:-len(u)//4 or None]) / max(1, len(u[len(u)//4:-len(u)//4 or None])), 1)}
json.dump(res, open(out + "/summary.json", "w")); print(json.dumps(res))
PY
