"""Full-benchmark wall-clock simulator: 2 settings x 50 tasks x 100 episodes on 8 GPUs.

Per-task step counts come from the public leaderboard success rates of the official π0.5 co-train
entry (robotwin_leaderboard.json) and the task step limits:
    steps/episode = p * succ_frac * limit + (1 - p) * limit
succ_frac (successful-episode length / limit) is the median seen in our smoke runs.

GPU model (fit to R05/R06 measurements): per step a job needs r seconds of the GPU's serial
rendering resource and c seconds of its own CPU. With k jobs on one GPU each job advances at
1 / max(r + c, k * r) steps/s (processor sharing of the render resource).
Fixed costs per episode and per job depend on the scheme.

usage: python makespan_sim.py robotwin_leaderboard.json
"""
import heapq
import json
import sys

LIMITS = {"adjust_bottle": 400, "beat_block_hammer": 400, "blocks_ranking_rgb": 1200, "blocks_ranking_size": 1200,
          "click_alarmclock": 400, "click_bell": 400, "dump_bin_bigbin": 600, "grab_roller": 400,
          "handover_block": 800, "handover_mic": 600, "lift_pot": 400, "move_can_pot": 400,
          "move_playingcard_away": 400, "move_stapler_pad": 400, "hanging_mug": 900, "open_laptop": 700,
          "open_microwave": 1500, "pick_diverse_bottles": 400, "pick_dual_bottles": 400, "place_a2b_left": 400,
          "place_a2b_right": 400, "place_bread_basket": 700, "place_bread_skillet": 500, "place_can_basket": 700,
          "place_cans_plasticbox": 800, "place_container_plate": 400, "place_dual_shoes": 600,
          "place_empty_cup": 500, "place_fan": 400, "place_burger_fries": 500, "place_mouse_pad": 400,
          "place_object_basket": 700, "place_object_scale": 400, "place_object_stand": 400,
          "place_phone_stand": 400, "move_pillbottle_pad": 400, "place_shoe": 500, "press_stapler": 400,
          "put_bottles_dustbin": 1700, "put_object_cabinet": 700, "rotate_qrcode": 400, "scan_object": 500,
          "shake_bottle": 700, "shake_bottle_horizontally": 700, "stack_blocks_three": 1200,
          "stack_blocks_two": 800, "stack_bowls_three": 1200, "stack_bowls_two": 900, "stamp_seal": 400,
          "turn_switch": 400}
ORDER = list(LIMITS)  # all_tasks.yml order (alphabetical-ish as listed there)
SUCC_FRAC = 0.35
EPISODES = 100


def norm(name):
    return name.lower().replace(" ", "_").replace("a2b", "a2b").replace("qrcode", "qrcode")


def jobs_from_leaderboard(path):
    d = json.load(open(path))
    e = next(x for x in d["entries"] if x["id"] == "cotrain:Pi_05")
    names = [norm(t) for t in d["tasks"]]
    jobs = []
    for setting, rates in (("clean", e["easy"]), ("randomized", e["hard"])):
        for n, p in zip(names, rates):
            p /= 100.0
            lim = LIMITS[n]
            spe = p * SUCC_FRAC * lim + (1 - p) * lim
            jobs.append({"task": n, "setting": setting, "p": p, "limit": lim, "steps": EPISODES * spe})
    return jobs


# scheme: per-step render r, cpu c (s); per-episode fixed (s, CPU-side); per-job fixed (s); slots/GPU
SCHEMES = {
    "official (1 job/GPU, all_tasks order)": dict(r=0.36, c=0.09, ep=8.0, job=60.0, k=1, lpt=False),
    "E0 render cuts, 1 job/GPU": dict(r=0.115, c=0.065, ep=8.0, job=60.0, k=1, lpt=False),
    "E0 full, 4 jobs/GPU, all_tasks order": dict(r=0.115, c=0.065, ep=0.6, job=3.0, k=4, lpt=False),
    "E0 full, 4 jobs/GPU, longest first": dict(r=0.115, c=0.065, ep=0.6, job=3.0, k=4, lpt=True),
    "E0 full, 2 jobs/GPU, longest first": dict(r=0.115, c=0.065, ep=0.6, job=3.0, k=2, lpt=True),
    # deferred video (R09): GPU ~4 ms/step (3 cams at chunk start + inference); CPU per step measured
    # 81 ms alone and 122-153 ms with 12 jobs on one GPU (R09b) -> use 0.14 s
    "deferred video, 13 jobs/GPU (all at once), longest first": dict(r=0.004, c=0.14, ep=0.6, job=3.0, k=13, lpt=True),
}


def simulate(jobs, r, c, ep, job, k, lpt, gpus=8):
    """Event-driven processor sharing. Fixed per-episode/per-job time is modelled as extra
    CPU-only work at rate 1 (it does not use the render resource)."""
    queue = sorted(jobs, key=lambda j: -j["steps"]) if lpt else list(jobs)
    # each running job: remaining render-steps and remaining fixed seconds
    running = {g: [] for g in range(gpus)}
    t = 0.0
    qi = 0
    finish = []

    def fill():
        nonlocal qi
        for g in range(gpus):
            while len(running[g]) < k and qi < len(queue):
                j = queue[qi]
                qi += 1
                running[g].append({"j": j, "steps": j["steps"], "fixed": job + EPISODES * ep})

    fill()
    while any(running.values()):
        # time to next completion on each GPU under current sharing
        best = None
        for g, lst in running.items():
            if not lst:
                continue
            n = len(lst)
            rate = 1.0 / max(r + c, n * r)
            for x in lst:
                dt = x["fixed"] + x["steps"] / rate
                if best is None or dt < best[0]:
                    best = (dt, g, x)
        dt = best[0]
        for g, lst in running.items():
            n = len(lst)
            if not n:
                continue
            rate = 1.0 / max(r + c, n * r)
            for x in lst:
                f = min(x["fixed"], dt)
                x["fixed"] -= f
                x["steps"] = max(0.0, x["steps"] - (dt - f) * rate)
        t += dt
        for g in running:
            done = [x for x in running[g] if x["steps"] <= 1e-6 and x["fixed"] <= 1e-6]
            for x in done:
                finish.append((t, x["j"]))
            running[g] = [x for x in running[g] if not (x["steps"] <= 1e-6 and x["fixed"] <= 1e-6)]
        fill()
    gpu_hours = sum(x["steps"] for x in jobs) * max(r, (r + c) / k) / 3600 + len(jobs) * (job + EPISODES * ep) / 3600 / k
    return t, finish


if __name__ == "__main__":
    jobs = jobs_from_leaderboard(sys.argv[1])
    tot = sum(j["steps"] for j in jobs)
    print(f"total steps {tot / 1e6:.2f} M over {len(jobs)} jobs")
    for j in sorted(jobs, key=lambda j: -j["steps"])[:6]:
        print(f"  longest: {j['task']}/{j['setting']} p={j['p']:.2f} limit={j['limit']} steps={j['steps']:.0f}")
    for name, s in SCHEMES.items():
        t, fin = simulate(jobs, **s)
        last = fin[-1][1]
        lower = max(tot * s["r"] / 8, max(j["steps"] for j in jobs) * (s["r"] + s["c"]))
        print(f"{name:42s} makespan {t / 3600:5.1f} h   (bound {lower / 3600:4.1f} h; last job {last['task']}/{last['setting']})")
