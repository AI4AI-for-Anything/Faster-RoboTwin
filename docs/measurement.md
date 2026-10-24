# Timing, the full-benchmark estimate, and why it stopped

## What is measured and what is estimated

| Number | Kind | Source |
|---|---|---|
| Per-episode equality of every fingerprint | measured, smoke scale | [validation](validation.md) |
| Per-step, per-episode and per-job costs | measured with timing wrappers, smoke scale (2–10 episodes per job) | rounds R00–R19 |
| Saturated per-GPU throughput of 6 jobs at 10–14 jobs per GPU | measured | [`measured_final.json`](../evalopt/data/measured_final.json) |
| Full benchmark: 73.5 h → 16.8 h, **4.38×** | **estimate** | `evalopt/full_estimate.py` |
| Ceiling with videos written during evaluation: about 4.6× | estimate | R12, R18, R19 |
| Deferred video: about 13–18× to scores | estimate, optional mode | R17, R18 |

No full 10,000-episode evaluation was run, neither official nor optimized.

## Timing

- Hardware: 8× H20 on a shared host. Other jobs ran on the same machine during most measurements, so timings carry
  load noise of about ±15% (the same baseline rollout took 172 s once and 200 s once). Speed ratios are therefore
  magnitudes; the bitwise conclusions are unaffected.
- Every arm starts its own policy server. Timing wrappers (`prof_eval.py`, `prof_server.py`) record per-call clocks
  grouped by context (`stats.json`), coarse events (`events.jsonl`) and per-episode start/end (`episodes.jsonl`).
- Two speed ratios are reported per case: rollout (`t_start` → `t_end` of each episode, without the expert check)
  and whole-job wall clock.
- Saturated throughput (`throughput_report.py`): the steady window runs from the moment the last worker has finished
  its first job's start-up to the moment the first worker has finished its job list, i.e. while all K workers are
  busy. Steps completed in the window (pro-rated per episode) divided by the window length.

## Per-task measurements behind the estimate

`measured_final.json`, built by `build_measured.py` from official baseline arms and saturated runs:

| Job | Official rollout s/step | Official s per accepted episode (expert check + setup) | Official episodes used | Faster steps/s per GPU | Jobs per GPU |
|---|---:|---:|---:|---:|---:|
| place_container_plate / clean | 0.4495 | 2.07 | 9 (clears included) | 9.76 | 14 |
| blocks_ranking_rgb / clean | 0.3875 | 5.72 | 1 | 9.80 | 10 |
| stack_blocks_three / clean | 0.3766 | 6.56 | 1 | 10.33 | 14 |
| open_laptop / randomized | 0.5815 | 4.07 | 7 (clears included) | 6.79 | 14 |
| place_dual_shoes / randomized | 0.5797 | 4.67 | 2 | 7.06 | 14 |
| handover_block / randomized | 0.4815 | 5.00 | 1 | 8.73 | 10 |

"Official episodes used" excludes each arm's first episode (its first-frame compile and first inference are one-off
costs, counted in the per-job start-up instead). Only the place_container_plate (10 episodes) and open_laptop
(8 episodes) official arms ran long enough to include SAPIEN's cache clear every 5 episodes; for the others the
estimate adds 2 s per episode. The handover_block throughput comes from the run in which one job hit the renderer
error described in [validation](validation.md#coverage).

## The estimate

`python evalopt/full_estimate.py evalopt/data/robotwin_leaderboard_pi05.json evalopt/data/measured_final.json`

- **Steps per job** (task × setting, 100 episodes): `100 × (p × 0.35 × limit + (1 − p) × limit)`, where `p` is the
  official π0.5 co-train success rate from the public leaderboard, `limit` the task's step limit, and 0.35 the
  successful-episode length relative to the limit seen in the smoke runs (30–40%). Failed episodes run to the limit.
  This gives 3.85 M steps (1.68 M clean, 2.17 M randomized); 67% of them are in failed episodes.
- **Official job time** = 115 s start-up + 100 × per-episode cost + steps × per-step cost. **Official schedule** =
  `eval_policy_multitask.py`: 1 job per GPU in task-list order, invoked once for the 50 clean jobs and then for the 50
  randomized jobs.
- **Faster job GPU time** = steps / saturated per-GPU throughput. **Faster schedule** = longest job first onto the
  least-loaded GPU, processor sharing per GPU, a lone job capped by its single-job speed (CPU 0.05 s per step), plus
  150 s of start-up.
- Jobs that were not measured use the mean of the measured jobs of the same setting (harmonic mean for throughput).
- Expert checks are replayed in the Faster column (record once per job; the record is policy-independent and
  reusable across checkpoints). The cost of a first recording run is not included and was not measured at scale.

| | GPU time | Wall clock, 8 GPUs | Speedup |
|---|---:|---:|---:|
| Official, list order (default) | 538.8 GPU-h | 73.5 h | 1.00× |
| Official, longest job first | 538.8 GPU-h | 68.2 h | 1.08× |
| **Faster-RoboTwin** | **127.9 GPU-h** | **16.8 h** | **4.38×** wall, 4.21× GPU time (4.06× against the official pipeline scheduled longest-first) |

Per setting, GPU time: clean 198.6 → 46.7 GPU-h (4.25×), randomized 340.1 → 80.9 GPU-h (4.21×).

**Uncertainty.** The six measured jobs cover 8% of the benchmark's steps; the other 94 jobs are extrapolated from
setting means. Official per-step costs come from 2–10-episode runs. Step counts depend on leaderboard success rates
and on the 0.35 length ratio. A full run is the only calibration; the recommended first check is a full 100-episode
job of a long task (put_bottles_dustbin or open_microwave) in both pipelines.

`evalopt/makespan_sim.py` is the earlier, simpler schedule model of R08/R09 (fixed per-step render and CPU costs). It
gave 66.9 h for the official pipeline and 15.9 h for the full E0 set at 4 jobs per GPU, longest first; the numbers in
the README come from `full_estimate.py`, which replaced it once per-task saturated throughputs were measured.

## Why it stopped near 4.4×

With the eval video written during evaluation, every step needs one ray-traced head frame. That frame is the official
video itself and cannot be skipped or approximated without changing the output.

- **One frame is about 80–130 ms of GPU work and does not overlap with other frames.** The head camera uses the
  ray-tracing shader with 32 samples per pixel and OIDN denoising; time scales with the sample count (about 2.3 ms per
  sample, 4.9 ms at 1 sample), denoising takes 1–2 ms, and the GPU runs at full clock (1980 MHz, 120–196 W of 500 W).
  H20 has no RT cores. With 1–14 render processes, 1–14 scenes in one process, or two threads in one process, a GPU
  delivers about 12–13 frames per second in total (clean scenes; about 8.5 for randomized scenes): processes time-share
  the GPU, they do not overlap.
- **The final configuration is at 96% of the per-step GPU demand.** On place_container_plate / clean, a step needs
  91.9 ms for the head frame, 4.7 ms for the wrist frames at chunk start and 1.8 ms for inference (amortized), i.e.
  98.4 ms, which allows 10.2 steps per second per GPU; 14 jobs per GPU measured 9.76.
- **5× would need a step shorter than the head frame alone.** The official pipeline needs 462 ms per step on this task
  (rollout plus per-episode costs at 1 job per GPU); 5× is 92.4 ms.
- Over the whole benchmark: at the static-scene frame-rate limits (12.5 frames/s clean, 8.5 randomized, 1.04 frames
  per step) the benchmark needs 112.5 GPU-h, 14.1 h on 8 GPUs (5.23×). Frames inside an evaluation cost more than in a
  static probe (13–19% in R12; R19 attributes it to image content, the arms and objects in view, not to motion), which
  puts the ceiling at about 4.6×. The remaining gap of the 4.38× estimate is job heads and tails and time slicing.

### Tried and not adopted

| Direction | Result |
|---|---|
| Remove the render sync after each physics substep (`EVALOPT_NO_SUBSTEP_RENDER`) | E0, saves 1.4 ms of CPU per step; no throughput change once the GPU is saturated (8.46 vs 8.51 and 6.19 vs 6.18 steps/s) |
| Reuse identical consecutive frames | failed episodes of 800 and 1200 steps had no identical consecutive frames |
| Several scenes per render process | no gain: one process with 2–14 scenes gave 11.93–12.38 frames/s per GPU, 2–14 separate processes 12.45–13.04 |
| Two render threads in one process | no overlap: 159.3 vs 159.4 ms per pair of frames, identical pixels |
| Compute the passive arm forces once per physics substep instead of about 7 times | E0 candidate, 6–9 ms CPU per step; not implemented (CPU is not the bottleneck at saturation) |
| Batched multi-scene rendering | would need restructuring RoboTwin's single-scene design; pixel equality not guaranteed; not attempted |
| Split one task's 100 episodes over several processes | E1 (changes the random-number draw order); not done |
| Leave the video out of the evaluation (deferred video) | scores unchanged, videos rebuilt afterwards; optional, not default |

## Deferred video (optional)

Without the head frame during evaluation, a step needs only the three policy cameras at chunk start and inference, so
the GPU is no longer the bottleneck (24–27% busy at 14 jobs per GPU); per-job CPU time is. Measured at 14 jobs per
GPU: 38.4 steps/s (place_container_plate / clean) and 29.7 steps/s (open_laptop / randomized), about 4× the
throughput with video. Applied to the benchmark's steps this is 32.4 GPU-h (16.6× less GPU time) and about
4.1–5.7 h of wall clock (13–18×), limited by the longest single job (put_bottles_dustbin / randomized). These are time
to scores; rebuilding all videos afterwards costs about as much GPU time as rendering them during evaluation, and
rebuilding only the videos of successful episodes covers about a third of the steps. This mode was not chosen as the
default.
