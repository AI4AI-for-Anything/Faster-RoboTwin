"""Full-benchmark estimate from measured per-task costs (official vs ours), GPU time and wall clock.

Per job (task x setting, 100 accepted episodes), steps from the public leaderboard success rate:
    steps = 100 * (p * SUCC_FRAC * limit + (1 - p) * limit)
Official job time (1 job per GPU, the official default):
    J_OFF + 100 * e_off + steps * s_off
  s_off: rollout seconds per step, from the official reference arms (includes sapien's clear_cache
         penalty when the arm ran >= 6 episodes; otherwise CLEAR_PEN per episode is added to e_off)
  e_off: per-episode fixed cost = expert play_once over all seeds tried + setup_demo calls (the
         process's first setup excluded) / accepted episodes
Ours: GPU-saturated per-GPU throughput thr (steps/s per GPU, K>=10 jobs per GPU, measured on that task).
Unmeasured jobs use the measured mean of the same setting (s_off, e_off arithmetic; thr harmonic).

Wall clock:
  official default : scripts/eval_policy_multitask.py, jobs_per_gpu 1, list order (all_tasks.yml),
                     each GPU takes the next job when free; invoked once for demo_clean (50 jobs) and
                     once for demo_randomized (50 jobs), one after the other
  official LPT     : same, longest job first (what a better scheduler would give the official pipeline)
  ours             : processor sharing on each GPU, LPT assignment to the least-loaded GPU, a job's
                     speed limited both by its GPU share and by its single-job speed 1/(g + c)
usage: python full_estimate.py robotwin_leaderboard.json measured.json
"""
import heapq
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from makespan_sim import LIMITS, SUCC_FRAC, norm  # noqa: E402

GPUS = 8
J_OFF = 115.0     # official per-job start-up: sim import + Sapien_TEST + cuRobo + first frame compile + server
CLEAR_PEN = 2.0   # s/episode, official clear_cache every 5 episodes (measured on place_container, 10 episodes)
CPU_STEP = 0.05   # s/step of CPU in our pipeline (take_action etc.), bounds a lone job's speed
STARTUP_OURS = 150.0  # s, worker + multi-session server start-up, paid once at t=0 (in parallel)

lb = json.load(open(sys.argv[1]))
meas = json.load(open(sys.argv[2]))
e = next(x for x in lb["entries"] if x["id"] == "cotrain:Pi_05")
names = [norm(t) for t in lb["tasks"]]
order = list(LIMITS)  # all_tasks.yml order


def fallback(setting):
    vs = [v for k, v in meas.items() if k.endswith(":" + setting)]
    return {"s_off": sum(v["s_off"] for v in vs) / len(vs),
            "e_off": sum(v["e_off"] for v in vs) / len(vs),
            "thr": len(vs) / sum(1 / v["thr"] for v in vs)}


jobs = []
for setting, rates in (("clean", e["easy"]), ("randomized", e["hard"])):
    rate = dict(zip(names, rates))
    fb = fallback(setting)
    for n in order:
        p = rate[n] / 100
        steps = 100 * (p * SUCC_FRAC * LIMITS[n] + (1 - p) * LIMITS[n])
        m = meas.get(f"{n}:{setting}", fb)
        e_off = m["e_off"] + (0.0 if m.get("clear_included") else CLEAR_PEN)
        jobs.append({"job": f"{n}:{setting}", "steps": steps, "measured": f"{n}:{setting}" in meas,
                     "t_off": J_OFF + 100 * e_off + steps * m["s_off"], "thr": m["thr"]})


def list_schedule(durations):
    free = [0.0] * GPUS
    heapq.heapify(free)
    end = 0.0
    for d in durations:
        t = heapq.heappop(free)
        heapq.heappush(free, t + d)
        end = max(end, t + d)
    return end


def ours_makespan():
    # assign jobs (longest GPU-time first) to the GPU with the least assigned GPU-time
    loads = [[] for _ in range(GPUS)]
    tot = [0.0] * GPUS
    for j in sorted(jobs, key=lambda j: -j["steps"] / j["thr"]):
        g = min(range(GPUS), key=lambda i: tot[i])
        loads[g].append(j)
        tot[g] += j["steps"] / j["thr"]
    end = 0.0
    for lst in loads:
        # processor sharing of the GPU (each job needs 1/thr GPU-seconds per step), job speed capped
        # by its lone speed 1/(1/thr + CPU_STEP)
        rem = {id(j): j["steps"] for j in lst}
        need = {id(j): 1 / j["thr"] for j in lst}
        t = 0.0
        while rem:
            n = len(rem)
            share = 1.0 / n
            speed = {k: min(share / need[k], 1 / (need[k] + CPU_STEP)) for k in rem}
            dt = min(rem[k] / speed[k] for k in rem)
            t += dt
            for k in list(rem):
                rem[k] -= speed[k] * dt
                if rem[k] <= 1e-6:
                    del rem[k]
        end = max(end, t)
    return end + STARTUP_OURS


off_gpu = sum(j["t_off"] for j in jobs)
our_gpu = sum(j["steps"] / j["thr"] for j in jobs) + STARTUP_OURS * GPUS
# the official scheduler is invoked once per task config: 50 clean jobs, then 50 randomized jobs
off_default = sum(list_schedule([j["t_off"] for j in jobs if j["job"].endswith(":" + s)]) for s in ("clean", "randomized"))
off_lpt = list_schedule(sorted((j["t_off"] for j in jobs), reverse=True))
ours = ours_makespan()
cov = sum(j["steps"] for j in jobs if j["measured"]) / sum(j["steps"] for j in jobs)
print(f"jobs {len(jobs)}, steps {sum(j['steps'] for j in jobs) / 1e6:.2f} M, measured tasks cover {cov * 100:.0f}% of steps")
print(f"GPU time      official {off_gpu / 3600:6.1f} GPU-h   ours {our_gpu / 3600:6.1f} GPU-h   -> {off_gpu / our_gpu:.2f}x")
print(f"wall (8 GPU)  official default order {off_default / 3600:5.1f} h   official LPT {off_lpt / 3600:5.1f} h   ours {ours / 3600:5.1f} h")
print(f"              speedup vs official default {off_default / ours:.2f}x   vs official LPT {off_lpt / ours:.2f}x")
for s in ("clean", "randomized"):
    o = sum(j["t_off"] for j in jobs if j["job"].endswith(s))
    u = sum(j["steps"] / j["thr"] for j in jobs if j["job"].endswith(s))
    print(f"  {s:10s} GPU time official {o / 3600:6.1f} h  ours {u / 3600:6.1f} h  {o / u:.2f}x")
