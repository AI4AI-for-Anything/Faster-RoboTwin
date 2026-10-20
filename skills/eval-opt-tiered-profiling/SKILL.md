---
name: eval-opt-tiered-profiling
description: Tiered profiling of closed-loop evaluation: job → episode (expert check / scene setup / rollout) → step (rendering / physics / TOPP / inference / communication) → camera; concurrency-scaling ablations to find the bottleneck; the three kinds of hotspots. Read before choosing what to optimize.
---

> Adapted from `train-opt-tiered-profiling` in [ResearchBoost-Agent](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/main/skills) for closed-loop policy evaluation. Measured on RoboTwin `ea8b211` + the official π0.5 co-train checkpoint on 8× H20; re-verify on your setup.

# Tiered profiling of closed-loop evaluation

## The layers of one evaluation job

| Layer | Contents | First-level timing (always on) | How to read it |
|---|---|---|---|
| Job start-up | Simulator process import, `Sapien_TEST`, cuRobo `MotionGen` build + warm-up (`set_planner` inside the first `setup_demo`); policy server import, checkpoint load, XLA compilation + autotune of the first inference | One-off wall clock | Paid once per task; officially every task restarts both processes |
| Per episode: expert check | `setup_demo` + `play_once` (cuRobo planning + physics) + `close_env` | events of `evalopt/prof_eval.py` | Independent of the policy; rejected seeds pay it too |
| Per episode: evaluation scene setup | Second `setup_demo` (new Engine/Renderer/Scene, load objects, 2500-step stability check) | Same | |
| Per step | `get_obs` (`_update_render` + `take_picture`/`get_picture` per camera), `take_action` (TOPP + about 87 physics substeps, each with `_update_render` + `check_success`), observation conversion, ws encode/decode, `update_obs`, one `get_action` per action chunk | stats of `evalopt/prof_eval.py` (grouped by context) | Failed episodes run to the step limit (400–1700 steps), so weight per-step costs by "failure rate × step limit" |
| Per camera | Wall clock and process CPU time of one camera render | `evalopt/render_bench.py` | |

`evalopt/prof_eval.py` only reads clocks and hashes data that already exists; it draws no random numbers and touches no simulator state. It wraps the original functions of the official `eval_policy_xpolicylab.py`. `evalopt/prof_server.py` does the same for the official server. **A profiled run must itself pass the per-episode comparison** (fingerprints identical to a reference without the profiler); otherwise it cannot be used to localize anything.

## Measured breakdown (adjust_bottle, clean, 2 episodes, official protocol, server and simulator on different GPUs)

| Item | Time | Share |
|---|---|---|
| Job start-up (simulator import 8.5 s + Sapien_TEST 2 s + cuRobo warm-up 21.5 s; server import 6 s + load 4.8 s + first-inference compilation 9 s) | about 52 s | once per task |
| `get_obs` rendering (241 calls, 0.39 s on average) | 94 s | over 80% of the rollout |
| `take_action` (physics 4.8 s, TOPP 2.3 s, check_success 0.9 s, etc.) | 12 s | |
| Expert check `play_once` | 6–8 s per episode | |
| Scene setup `setup_demo` | 0.6 s × 2 per episode | |
| Inference (steady state) | 86 ms per call × 3 calls per episode | small |
| ws encode/decode, observation conversion, `update_obs` | about 1.5 s in total | small |

Single camera (320×240, rt shader 32 spp + OIDN): about 50 ms for each wrist camera, about 70 ms for the head camera, about 90 ms for the front camera; one `get_obs` about 270 ms. **The front camera feeds neither the policy nor the video**, so about 1/3 of every step's rendering is wasted.

**The policy reads an observation only once, at the start of each action chunk** (50 steps per chunk), but the official loop renders all 4 cameras and calls `update_obs` at every step; the observations of the 49 steps in between are overwritten by the server, and only the head camera's frame goes into the eval video.

## Finding the bottleneck: concurrency-scaling ablation

nvidia-smi utilization says nothing here: with one simulator process the GPU shows only about 25% utilization, which does not mean more processes fit.

Ablation: start K identical simulator processes on the same GPU at once (`scripts/evalopt/probes/scale_test.sh`). Measured on H20:

| K | Median `get_obs` per process | Throughput of the GPU |
|---|---|---|
| 1 | 272 ms | 3.7 per s |
| 2 | 539 ms | 3.7 per s |
| 4 | 1066 ms | 3.75 per s |
| 8 | 2134 ms | 3.75 per s |

Throughput does not grow with the number of processes at all: **rendering is a serial resource of one GPU** (the processes' Vulkan work is time-shared on the GPU, not parallel; each waiting process spins on the CPU, cpu/wall≈1). H20 is a Hopper GPU without RT cores, so ray tracing is expensive. Conclusions:

- More simulator processes per GPU (official `jobs_per_gpu>1`) is not faster; it only slows every task down.
- The only lossless lever on rendering is **rendering less**: do not render cameras nobody reads, and do not render frames nobody reads.
- Physics/TOPP (`take_action`, about 44 ms per step) is not affected by concurrency on the same GPU; it is a CPU resource.

## Three kinds of hotspots

1. **Wasted work** (can be removed without changing any number): the front camera; the wrist cameras and `update_obs` at mid-chunk steps; server start-up and compilation repeated per task; the policy-independent expert check redone in every run.
2. **Configuration/scheduling problems**: non-reproducible autotune (fix first, otherwise E0 cannot be judged); the official batch path cross-talks for stateful policies (unusable); interference between server and simulator on the same GPU (to be measured).
3. **Real computation**: one head-camera frame per step (needed for the eval video), physics, TOPP, inference. These numbers may not change; they can only be spread over more GPUs and a better schedule.

## Rules

- Account by layer first, then pick a hotspot; profile the current best version every round.
- For conclusions about timings that depend strongly on rendering, state what else was running on the same GPU and host at the time.
- Step limits and failure rates differ a lot between tasks, so ratios from one short task cannot be extrapolated to 50 tasks; full-benchmark estimates must use "steps per task × per-step cost + per-episode fixed cost + per-task start-up".
