# Validation: every episode bitwise identical

An evaluation is a measurement of the policy, so the evaluation pipeline may not add any numerical freedom. Every
adopted switch was judged at level **E0**: each episode of the candidate matches the official baseline bit for bit.

## Levels

| Level | Meaning | Example | Decision |
|---|---|---|---|
| E0 bitwise | every per-episode fingerprint equals the baseline | skipping renders nobody reads, resident server reset to `key(0)` per job, task-level concurrency, replayed expert checks | judged by `compare_arms.py` |
| E1 same distribution | each inference computes the same numbers, but the random-number draw order changes | splitting one task's 100 episodes over several processes | not done; needs a user decision |
| E2 numbers change | inference or rendering values change | cross-environment batched inference, fewer samples per pixel, other denoiser, fewer flow steps, lower precision | not done |
| E3 protocol change | what is measured changes | fewer episodes, no expert check, other seeds or step limits, no eval video | not done; the deferred-video mode only moves video production and is optional |

## Fingerprints

`evalopt/prof_eval.py` hashes data that is already sent or returned (it draws no random numbers and does not touch
simulator state) and writes one record per episode to `episodes.jsonl`. `evalopt/compare_arms.py` (two arms in one
case directory) and `evalopt/compare_dirs.py` (two job directories) compare, per episode:

- `seed`, `instruction`, `success`, `steps`, `n_get_action` (number of policy queries);
- `consumed_obs_sha`: for every `get_action`, the hash of the observation the policy actually consumed (three RGB
  images, all state entries and the instruction);
- `chunk_sha`: the hash of every action chunk the policy returned;
- `video_sha`, `video_frames`: the hash of every frame piped to the official ffmpeg process, and the frame count.

A run is compared only if it is valid (below). Equal success rates are not enough: two runs with the same success
rate can already have different actions.

<a id="reference-and-candidate"></a>

## Baseline and candidate

- **Baseline** = the official functions plus the timing and hashing wrappers. **Candidate** = the same plus
  switches. Each arm starts its own policy server (the JAX random-number chain starts at `key(0)`, as for one official
  job), with server and simulator on the same GPU, as the official `eval.sh` places them.
  `scripts/evalopt/run_pair.sh` runs baseline → candidate (→ second baseline with `REPEAT_REF=1`).
- Both arms load the same pinned XLA autotune file, and, where expert checks are replayed, the same record.
- **The wrappers are transparent:** the same case run through the plain official entry
  (`scripts/eval_policy.sh` with the unmodified server, `run_plain.sh`) produced eval videos whose mp4 files had the
  same md5 as the wrapped baseline arm and the optimized arm (adjust_bottle / clean, 3 episodes; `_result.txt` 1.0).

<a id="making-the-reference-reproducible"></a>

## Making the baseline reproducible

The official pipeline is not bitwise reproducible from run to run. Three sources were found and handled:

1. **XLA autotune.** On the first inference, XLA times GEMM and convolution implementations and keeps the fastest.
   Under load the choices change and so do the actions: with two render processes and 48 busy CPU loops, one of two
   server starts chose 8 of 25 entries differently and returned different actions for the same observation and noise;
   full episodes with the same seed ran 125/114 instead of 123/118 steps. The official 8-GPU schedule autotunes under
   load, so official results already differ between runs. Fix: record the autotune results once on an idle GPU and load
   them in every server (`--xla_gpu_load_autotune_results_from`, file in `evalopt/autotune/`). This fixes one of the
   baseline's own legal choices and does not change precision. Re-record it for another GPU model, driver or JAX
   version (`--xla_gpu_dump_autotune_results_to=<file>` on an idle GPU).
2. **Expert check across processes.** For place_dual_shoes / randomized seed 100001, four identical processes (same
   inputs, same call order, policy replaced by a client that ends each episode at once) accepted the seed twice and
   rejected it twice (planning succeeded each time; `check_success` after execution differed). So *which* 100 seeds
   the official protocol evaluates can change between runs. Fix: record one realization of the expert check
   (`EVALOPT_EXPERT_RECORD`) and let both arms replay it (`EVALOPT_EXPERT_REPLAY`). The record is a legal realization
   of the official filter and does not depend on the policy. cuRobo also keeps in-process history (its Halton sampler
   advances), so a second expert plan for the same seed in the same process differs; the evaluation episode itself
   does not depend on whether an expert demonstration ran before it (first frame and the frame after 30 steps were
   identical).
3. **Instructions depend on `--test_num`.** `generate_episode_descriptions` builds a candidate list whose length is
   `test_num` and picks one entry with `Random(f"{seed}:select")`, so the same seed gets different instructions in a
   2-episode smoke run and in the 100-episode protocol. Smoke runs are not subsets of the formal run, and a job may not
   be split without keeping the `test_num=100` semantics.

With the autotune pinned, two references of adjust_bottle / clean (separate servers and simulator processes) were
bitwise identical.

## Why the render cuts are E0

Skipping a render is E0 only if rendering does not change physics, does not draw random numbers used elsewhere, and a
frame's pixels depend only on the current scene. Measured on the RoboTwin ray-tracing shader (32 samples per pixel,
depth 8, OIDN denoising) on H20:

- With `_update_render()` called before each picture, re-rendering the same scene gives identical pixels; taking a
  second picture **without** `_update_render()` changes the pixels. The switches therefore never skip
  `_update_render()` (it also draws `np.random` when random lighting is on).
- Rendering all cameras every step and rendering only at the last step give an identical last frame and an identical
  physics-state sequence, also after 25 static steps.
- 1–8 processes rendering concurrently on one GPU produce identical pixels.
- `Pi_05.Model` keeps only the latest observation (`observation_window` is overwritten, no history), and
  `get_action` reads only that. The observation sent before each `get_action` is always the full one from the top of
  the chunk loop. This premise has to be re-checked for a policy with an observation history.

The policy never receives fewer cameras: at every chunk start the head and both wrist cameras are rendered and sent
as before. What is skipped is the front camera (used by neither the policy nor the video) and the two wrist cameras
plus `update_obs` on the 49 steps inside a chunk, whose observations the server would overwrite before the next query.

## Valid runs

Exit code 0 is not success. A job is invalid, and is rerun rather than compared or counted, if:

- a log contains `Failed to generate episode instruction`, `Policy rollout error`, `worker_error` or
  `env error before first action` (the official loop counts a rollout exception as a failed episode);
- the number of episodes differs from `--test_num`, or the sequence of skipped seeds differs from the baseline;
- the server recompiles a shape after the first inference, or autotunes after the first inference;
- the run fingerprint (command line, configs, checkpoint, autotune file, code revision, visible GPUs) differs from the
  baseline.

## Coverage

All comparisons are smoke-scale (2–10 episodes per job), not scores. Seven tasks were compared: adjust_bottle,
blocks_ranking_rgb, place_container_plate, stack_blocks_three (clean) and handover_block, open_laptop,
place_dual_shoes (randomized), including failed episodes that ran to their step limit (800 and 1200 steps).

| Round | What was compared | Result |
|---|---|---|
| R02 | skipped renders, 5 cases, 1 job per GPU | all episodes equal (place_dual_shoes: episode 1 equal, episode 2 evaluated another seed because of the expert-check non-reproducibility, resolved in R04) |
| R04 | expert record → replay, 3 cases (with seeds filtered by expert errors) | all equal |
| R06 | 4 and 6 jobs per GPU, memory share 0.08 | 10 / 10 jobs |
| R07 | resident server and simulator, lazy cuRobo | 8 / 8 jobs |
| R08 | full set at 1 and 4 jobs per GPU | 5 / 5 jobs |
| R10 | substep render sync removed (not adopted), 6 jobs per GPU | 24 / 24 jobs |
| R10 | blocking CUDA sync, frame by frame | 53 / 53 frames |
| R11–R12 | multi-session server, blocking sync, kept cache, 6–14 jobs per GPU | 136 jobs equal; one more job invalid (renderer error, below) |
| R12 | kept render cache vs. official clears every 5 episodes, 10- and 8-episode runs | equal |
| R16 | resident worker, 12 episodes with kept cache | 4 / 4 jobs |
| R19 | frame-log runs | 5 / 5 jobs |

In the 10-jobs-per-GPU handover_block / randomized run, one job's second episode hit `ErrorOutOfHostMemory` at step
376 while about 65 render processes ran on the host; the official loop recorded it as a failure after 376 instead of
800 steps. That job is invalid under the rule above (renderer error, not a switch effect), and no rerun of it is
recorded. The other 19 jobs of that run were equal.

Not covered: the two longest tasks (open_microwave and put_bottles_dustbin, step limits 1500–1700) were not compared;
the 100-episode protocol was not run end to end with or without the switches.

## Deferred video

`EVALOPT_NO_VIDEO=1` changes when videos are produced, so it is optional and not part of the default configuration.

- Scores and every fingerprint except the video were identical to the official baseline in the two R09 cases
  (6 episodes), in 25 jobs at 1–12 jobs per GPU (R09b) and in 84 jobs at 14 jobs per GPU (R17a).
- Videos rebuilt by `replay_video.py` had identical frame hashes and frame counts, and mp4 files with the same md5 as
  the official eval videos on 6 episodes: place_container_plate / clean (137, 139, 146 steps, all successful) and
  place_dual_shoes / randomized (238 and 208 steps successful, one failure at the 600-step limit).

## Tooling

| Script | Purpose |
|---|---|
| `evalopt/compare_arms.py <case_dir>` | per-episode judge of `ref` against every other arm in a case directory, plus wall and rollout speedups |
| `evalopt/compare_dirs.py <ref> <cand>` | the same judge for two job directories |
| `evalopt/throughput_report.py <run> [<ref>...]` | steady-state per-GPU throughput of a multi-worker run, with bitwise checks against baseline jobs |
| `evalopt/expert_rep_report.py a.pkl b.pkl ...` | seed-by-seed comparison of expert-check records |
| `evalopt/infer_det.py`, `evalopt/expert_det.py`, `evalopt/render_static.py`, `evalopt/render_bench.py` | reproducibility probes behind the findings above |
