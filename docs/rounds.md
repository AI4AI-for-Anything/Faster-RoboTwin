# Optimization rounds R00–R19

The optimization ran on 2026-10-17/18 on 8× H20. Each round profiled the current best version, changed one thing
behind a switch, and judged it per episode against the official baseline before looking at speed. All numbers are
smoke scale (2–10 episodes per job) and are not scores. Rounds R13–R15 and R17 are reported together with R12 and
R16. Raw artifacts (logs, videos, per-run directories) are not included.

| Round | Topic | Outcome |
|---|---|---|
| [R00](#r00-baseline-profile) | baseline profile of the official pipeline | rendering is 80%+ of the rollout; the front camera and mid-chunk wrist renders are never read |
| [R01](#r01-baseline-reproducibility) | reproducibility of the baseline | autotune changes under load → pinned; rendering is stateless; evaluation episodes do not depend on expert history |
| [R02](#r02-skip-unread-renders) | skip unread renders | E0; rollout 2.5–2.8× |
| [R03](#r03-concurrency) | jobs per GPU | official: rendering is serial per GPU, more jobs do not help; after R02: 4 jobs per GPU 1.83× |
| [R04](#r04-expert-check-record-and-replay) | expert-check record / replay | the official expert check is not reproducible across processes; replay makes whole tasks comparable and removes its cost |
| [R05](#r05-profile-after-render-cuts) | profile after the render cuts | head camera 58–69%, take_action 23–28%, about 60 s fixed per job |
| [R06](#r06-several-jobs-per-gpu-official-placement) | several jobs per GPU with server and simulator together | memory share 0.08 is E0; 4 jobs per GPU 2.15× batch throughput |
| [R07](#r07-shared-start-up) | resident server and simulator, lazy cuRobo | E0 on 8 jobs; fixed cost per job 45–60 s → 2–3 s from the 2nd job |
| [R08](#r08-profile-after-the-full-e0-set) | profile after the full E0 set | a head frame is ~80 ms of pure ray tracing; frames do not overlap; longest-first scheduling simulated at 15.9 h |
| [R09](#r09-deferred-video) | deferred video | scores unchanged; rebuilt mp4 md5-identical (6 episodes); a user decision |
| [R10](#r10-what-limits-a-saturated-gpu) | what limits a saturated GPU | separate servers waste GPU time; cache clears cost 13–40 s of CPU; CUDA spin waits burn CPU |
| [R11](#r11-multi-session-server-and-blocking-sync) | multi-session server + blocking CUDA sync | E0; 6 jobs per GPU 4.84 → 8.46 steps/s |
| [R12](#r12r15-long-official-runs-saturation-and-the-frame-rate-ceiling) | long official runs, saturation per task, frame-rate ceiling | keep cache is E0; ceiling about 4.6× with video |
| [R16](#r16r17-resident-process-resources-and-14-jobs-per-gpu) | resident-process resources; 14 jobs per GPU | no leak; deferred video 30–38 steps/s per GPU |
| [R18](#r18-can-it-reach-5) | can it reach 5×? | 4.38× estimated with video; 5× not reachable; deferred video 13–18× |
| [R19](#r19-is-any-gpu-time-wasted) | is any GPU time wasted? | no: 96% of the per-step GPU demand |

## R00 Baseline profile

The official checkpoint ran through the official loop (adjust_bottle / clean, 2 episodes, 2/2 successful,
instructions generated normally). Of a 168 s job: `get_obs` 94.1 s over 241 steps (picture taking 92.2 s), cuRobo
planner build and warm-up 21.5 s, expert `play_once` 13.6 s, `take_action` 12.2 s, first inference with XLA compile
and autotune 9.0 s (then 86 ms per call), simulator import and `Sapien_TEST` 10.5 s, server import and load 10.9 s,
everything else about 1.5 s. Per camera (320×240, 32 samples per pixel + OIDN): wrists about 50 ms each, head about
70 ms, front about 90 ms. The front camera is mapped by neither the policy client nor the video; π0.5 reads one
observation per 50-step chunk, but the loop renders all four cameras and calls `update_obs` every step. A repeat of
the same configuration produced different step counts (125/114 vs 123/118), which led to R01.

<a id="r01-reference-reproducibility"></a>

## R01 Baseline reproducibility

Inference with a fixed observation and noise is bitwise reproducible across idle processes, but XLA autotune choices
vary with load: idle runs differed in 3 of 25 entries with identical outputs, loaded runs in 7–8 entries, and one
loaded run returned different actions. Pinning one autotune file recorded on an idle GPU fixed this (and cut the first
inference from 9 s to 6 s). Rendering is a pure function of the scene as long as `_update_render()` runs before each
picture; skipping or reordering renders does not change the last frame or the physics. Evaluation episodes do not
depend on whether an expert demonstration ran before them; a second expert plan of the same seed in one process
differs (cuRobo history).

## R02 Skip unread renders

`EVALOPT_SKIP_UNUSED_CAMS` and `EVALOPT_LAZY_OBS` (a copy of the official `eval_remote_policy` with one marked change).
All five cases matched bitwise, including failed episodes of 800 and 1200 steps; the second baseline of
adjust_bottle matched too. Rollout speedups 2.61–2.80×, job wall 2.05–2.38× (start-up and expert checks unchanged).
`get_obs` went from 0.47 to 0.13 s per step, `update_obs` calls per episode from hundreds to 3–19. The plain
official entry produced mp4 files with the same md5 as the wrapped arms. One case (place_dual_shoes) evaluated a
different seed in its second episode because the expert check accepted a seed in one arm and rejected it in the other;
R04 traced this to the official expert check. Lesson: a launcher script was overwritten while running and re-ran a
finished arm; scripts are now replaced by `mv` only.

## R03 Concurrency

With the official pipeline, K identical simulator processes on one GPU give a constant 3.7 `get_obs` per second for
K = 1, 2, 4, 8 (each process slows down linearly, pixels identical): rendering time-shares the GPU. After R02,
K jobs per GPU scale throughput 1.31× (K = 2) and 1.83× (K = 4), all bitwise identical to K = 1, because CPU work
becomes a larger share. Four 29 GB servers do not fit next to their simulators; JAX memory shares 0.3, 0.12 and 0.08
produced identical actions for the same input. The official `num_workers > 1` batch path mixes observations of
different workers in one stateful model and is not usable for π0.5.

## R04 Expert-check record and replay

Four identical processes (dummy policy) accepted seed 100001 of place_dual_shoes / randomized twice and rejected it
twice: the official seed selection is not reproducible across processes. `expert_cache.py` records per seed how the
expert phase ended and the `play_once` info, and replays it without simulation. place_container_plate (with two seeds
filtered by expert errors), open_laptop and place_dual_shoes then matched bitwise end to end; the expert phase cost
went to zero (rollout 2.47–2.82×, job 2.04–2.46×). Also found: instructions depend on `--test_num`. Lesson: a first
record run of open_laptop filtered every seed because the wrapper did not forward the `target=` keyword of
`check_success`; wrappers now forward all arguments.

## R05 Profile after the render cuts

Per step (1 job per GPU): 140–223 ms, of which the head camera (video only) 58–69%, the three policy cameras at chunk
start about 7 ms, `take_action` 23–28% (physics 12–22 ms, TOPP 9–14, setting joints and grippers 8–13, per-substep
render sync and success check 3–9), inference 5–26 ms including the amortized first compile (steady 83–86 ms per
call). Fixed per job: simulator import 14 s (6.6 s of it two `nvidia-smi` calls at sapien import), `Sapien_TEST`
3.6–3.9 s, cuRobo build and warm-up 23.5–26.7 s, first renderer 3.4 s, server 6 + 4.8 + 5.8 s. cProfile found the
passive arm forces computed about 7 times per physics substep.

## R06 Several jobs per GPU, official placement

Server and simulators on the same GPU with memory share 0.08: K = 1 5.5 steps/s, K = 4 8.6 (batch throughput
including start-up 2.15×), K = 6 9.1 (2.39×); 10 jobs bitwise identical. K = 4 already reaches the head-camera limit
of about 8.8 steps/s. Lesson: on a shared host a chosen port was already used by another service; the readiness check
passed against the wrong listener. Launchers now require a free port and a listener owned by the current user.

## R07 Shared start-up

`prof_eval.py --jobs` (one simulator process runs many jobs), `EVALOPT_SERVER_NEW_JOB` (reset a reused server to
`key(0)`), `EVALOPT_LAZY_CUROBO` (build cuRobo on first use; `plan_grippers` is a pure function). Eight jobs matched
their single-process references. From the second job, fixed cost dropped from 45–60 s to 2–3 s; four smoke jobs took
614 s instead of 820 s (1.34×). Over the full benchmark this part is about 1% of the GPU time.

## R08 Profile after the full E0 set

A head frame is 76–80 ms of GPU ray tracing (time proportional to samples per pixel, OIDN 1–2 ms, full clock, not
power-limited); two threads or several processes do not overlap frames. At 4 jobs per GPU, a GPU runs 8.6 steps/s
while all jobs are busy but only 7.0 on average when short jobs finish early, so jobs must be fed from a global queue.
A schedule simulation (`makespan_sim.py`) gave 66.9 h official, 19.0 h for the full set in task-list order and
15.9 h longest first, against a render bound of 15.4 h. Removing the substep render sync was E0 but gained only 1–2%.

## R09 Deferred video

`EVALOPT_NO_VIDEO` + `EVALOPT_RECORD_ACTIONS`, and `replay_video.py` to rebuild the videos. Scores and all other
fingerprints unchanged; rebuilt mp4 files md5-identical to the official ones (6 episodes, including a 600-step
failure). 25 more jobs at 1–12 jobs per GPU matched. Per step 75–80 ms without the head frame; throughput per GPU
about 4.6× that with video. It changes when videos are produced, so it was left to the user.

## R10 What limits a saturated GPU

With one server per job, 6 jobs per GPU reached only 4.84–5.42 steps/s: separate servers take GPU time slices and
memory. The official cache clear every 5 episodes costs 13–40 s of CPU before the next first frame (earlier 2–3
episode baselines never triggered it). CUDA waits spin, so each 76 ms frame also used 76 ms of CPU; blocking sync
reduced that to 0.8 ms (53 frames identical). Removing the substep render sync gave no throughput gain at saturation;
failed episodes had no identical consecutive frames to reuse.

## R11 Multi-session server and blocking sync

`multisession_server.py`: one server per GPU, one session per `evaluation_id` (shallow copy of the loaded model with
its own observation buffer and `key(0)` chain; parameters and jitted functions shared; calls serialized by the
official lock). With blocking sync and kept cache: 8.46 / 8.56 / 9.00 steps/s at 6 / 8 / 10 jobs per GPU
(GPU 90–93% busy, 44 GB at 10 jobs); all jobs bitwise identical.

## R12–R15 Long official runs, saturation and the frame-rate ceiling

Official runs of 10 and 8 episodes (with cache clears) cost 0.45–0.58 s per step, more than the short baselines had
shown. The full E0 set matched them bitwise (keep cache is E0) at 3.20× and 2.52× as single jobs. Saturated
throughput at 10–14 jobs per GPU: 9.76 (place_container_plate), 9.80 (blocks_ranking_rgb), 10.33
(stack_blocks_three), 6.79 (open_laptop), 7.06 (place_dual_shoes), 8.73 (handover_block) steps/s. One handover_block
job hit a renderer error under host-wide load (see [validation](validation.md#coverage)), and its worker later hung as a
zombie. A static scene renders at most about 12.0–12.7 frames/s per GPU (clean) and 8.2–8.6 (randomized). The first
full estimate gave 4.37×.

## R16–R17 Resident-process resources and 14 jobs per GPU

A resident worker running 12 episodes kept file descriptors, memory maps, threads and RSS flat, with and without
cache clears. Deferred video at 14 jobs per GPU: 38.4 and 29.7 steps/s, GPU 24–27% busy, 84 of 84 jobs matching in
all fields except the video.

## R18 Can it reach 5×?

With the video written during evaluation: estimated 73.5 → 16.8 h (4.38×, GPU time 4.21×); the frame-rate ceiling
is about 4.6×, so 5× is not reachable under this constraint. With deferred video: 32.4 GPU-h, about 13–18× to
scores; the user has not chosen it.

## R19 Is any GPU time wasted?

On a quiet host, total frame rate per GPU grew slightly from 1 to 14 render processes (11.98 → 13.04) and did not grow
with more scenes per process; scene motion does not change frame cost (78.8 vs 79.8 ms clean, 119.4 vs 119.9 ms
randomized). A per-frame log of real evaluations put the GPU demand at 98 ms per step (head 92, wrist 4.7, inference
1.8) on place_container_plate; 14 jobs per GPU reached 96% of it. No waste remains with the video on. Lesson: starting
70 render processes at once made every frame 5× slower, so workers must start staggered.
