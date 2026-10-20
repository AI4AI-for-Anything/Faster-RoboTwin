---
name: eval-opt-action-ladder
description: The action ladder for closed-loop evaluation optimization (L0 scheduling and resident processes → L1 removing wasted work → L2 caching policy-independent results → L3 changing the evaluation loop), each step with its E tier and measured evidence, plus what is explicitly not done. Read when deciding what to try this round.
---

> Adapted from `train-opt-action-ladder` in [ResearchBoost-Agent](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/main/skills) for closed-loop policy evaluation. Measured on RoboTwin `ea8b211` + the official π0.5 co-train checkpoint on 8× H20; re-verify on your setup. Same principle as the original: reuse first, remove wasted work first, write new code last; every action behind a switch and passed through the per-episode comparison on its own. Tier definitions are in `eval-opt-equivalence`.

# Evaluation optimization action ladder

## L0 Scheduling, resident processes, reproducibility

| Action | Tier | Evidence (RoboTwin + π0.5, 8× H20) |
|---|---|---|
| Pin XLA autotune (`--xla_gpu_load_autotune_results_from`) | E0 (makes the reference reproducible; the precondition of every bitwise judgement) | Without pinning, actions change under load (R01); with pinning, the first inference drops from 9 s to 6 s |
| Resident policy server, `evalopt_new_job` at job start (random-number chain reset to `key(0)` + observation cleared) | E0 | R07: 8 jobs bitwise identical; server import + load + compile (about 17 s) paid only once |
| Resident simulator process running several jobs (`evalopt/prof_eval.py --jobs`) | E0 | R07: from the 2nd job on, fixed cost 45–60 s → 2–3 s |
| Lazy cuRobo build (`EVALOPT_LAZY_CUROBO`; `plan_grippers` is a pure function and passes through) | E0 (with replayed expert checks cuRobo is never built) | R07b: 4 jobs bitwise identical, `setup_demo` about 30 s → 2.5 s |
| Server memory share 0.08 (`EVALOPT_MEM_FRACTION`) | E0 (only changes the allocator pool size) | R03 at unit level + R06 end to end, 10 jobs bitwise identical |
| Several jobs per GPU (server and simulator on the same GPU) | E0 | No gain in the official flow (rendering is serial, R03); after the render cuts K=4 gives 2.15× batch, K=6 2.39×, and the rollout reaches the render limit at K=4 (R06) → 4 jobs per GPU |
| Task-level scheduling: global queue, longest expected job first | E0 | Jobs do not influence each other (once autotune is pinned and expert checks are replayed) |
| JAX persistent compilation cache | To be measured | Little value once the server is resident |

## L1 Removing wasted work

| Action | Tier | Evidence |
|---|---|---|
| Do not render cameras that neither the policy nor the video reads (front) | E0 | Render statelessness measured; saves about 90 of 270 ms per step |
| Mid-chunk steps render only the head camera (needed by the video), not the two wrist cameras, and send no `update_obs`; the chunk start does the full `get_obs` + `update_obs` + `get_action` as usual | E0 (premise: the policy reads only the latest observation; not a single `_update_render()` is skipped) | Rendering per step drops from 4 cameras to about 1.06 |
| Do not write the eval video | E3 (success rate unchanged, but an official artifact is missing) | User decision; after the render cuts the head camera takes 58–69% of every step (R05) |
| Move the video off the critical path: no head camera during evaluation (`EVALOPT_NO_VIDEO` + `EVALOPT_RECORD_ACTIONS`), rebuild the video afterwards with `evalopt/replay_video.py` | Evaluation results E0; the video is a change in artifact flow, user decision | R09: mp4 rebuilt for 6 episodes md5-identical to the official ones; about 4.6× throughput on one GPU; the bottleneck becomes the CPU of the longest single job |
| Multi-session policy server (`evalopt/multisession_server.py`: one session per evaluation_id, one copy of the parameters, per-session observation and `key(0)` random-number chain) | E0 | R11: 136 jobs bitwise identical; K=6 4.84 → 8.46 steps/s; 44 GB of GPU memory for 10 jobs per GPU |
| CUDA blocking sync (`EVALOPT_CUDA_BLOCKING_SYNC`) | E0 (only changes how the CPU waits) | CPU per frame 76.9 → 0.8 ms, pixels bitwise identical |
| Do not clear the render cache (`EVALOPT_KEEP_CACHE`) | E0 | Long runs bitwise identical; 12 episodes 1028 → 757 s; no resource growth |
| Compute the passive forces of both arms once per physics substep (`_entity_qf`, currently about 7 times) | E0 candidate (state does not change within a substep) | About 6–9 ms per step, CPU, small gain |

## L2 Caching policy-independent results

| Action | Tier | Prior |
|---|---|---|
| Record the expert-check result of every seed (outcome + the episode info returned by `play_once`) once (`EVALOPT_EXPERT_RECORD`), then replay it in every later evaluation, also for other checkpoints (`EVALOPT_EXPERT_REPLAY`) | E0 relative to "the official flow replaying the same record"; bitwise identity to "the official flow running the expert check live" is impossible, because the official flow is not reproducible across processes itself (see the equivalence card) | Both a speedup (saves 4–8 s per episode and the whole cost of rejected seeds) and the precondition of a bitwise judgement; bind the record to code revision, task, configuration and seed start |
| Scene asset cache (SAPIEN mesh/texture cache, officially cleared every 5 episodes) | To be measured (does clearing change pixels?) | |

## L3 Changing the evaluation loop

- A change must be a minimal-diff copy of the official function (see the diff command at the top of `evalopt/opt_loop.py`), with an `EVALOPT` comment at every changed spot, behind an environment-variable switch that is off by default.
- Compare every added change with the reference on its own, then compare the combination again.

## E1: needs the user's decision

| Action | Why it is not E0 |
|---|---|
| Split one task's 100 episodes across several processes | The policy's random-number chain accumulates per process; cuRobo's in-process history affects the expert check → the success rate is another sample from the same distribution |
| Use the official `num_workers>1` batch path | For π0.5 not even E1: several clients sharing one stateful model cross-talk, which is wrong |

## Not done (E2/E3)

- Cross-environment batched inference: the GEMM shape changes, so does the kernel, and with it the rounding of the actions. Inference is only a few percent anyway.
- Lower spp, changed/disabled denoiser, another shader, another resolution, rendering on another GPU model, changed flow steps, lower precision.
- Fewer episodes, expert_check off, different seed start, different step limit, fixed instructions.

## Symptom → first suspect

| Symptom | First suspect |
|---|---|
| Same seed, same configuration, two runs with different step counts | Autotune not pinned; load on the host |
| The candidate is slower per step than the reference | Another render process on the same GPU (rendering is time-shared) |
| An episode's instruction became the task name | Silent fallback in `build_instruction` |
| The candidate mismatches only from the 2nd episode on | Random-number chain or in-process history (server not restarted/reset, different expert-check order) |
| Only the video hash mismatches | Missing `_update_render()` before the head-camera frame, or the last frame of the success branch |
