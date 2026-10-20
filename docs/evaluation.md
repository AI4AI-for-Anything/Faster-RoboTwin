# Evaluation setup, launch and switches

Faster-RoboTwin keeps the official RoboTwin 2.0 evaluation of the official π0.5 co-train checkpoint and adds
`EVALOPT_*` switches around it. Upstream RoboTwin and XPolicyLab files are not modified: the switches live in
wrappers (`evalopt/prof_eval.py`, `evalopt/prof_server.py`, `evalopt/multisession_server.py`) and in two
minimal-difference copies of upstream functions (`evalopt/opt_loop.py`, `evalopt/opt_take_action.py`). With no
switch set, the wrappers call the official functions unchanged.

## Sources

| Component | Version |
|---|---|
| RoboTwin | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) `ea8b211` (this repository's base) |
| XPolicyLab | submodule `fa431ec` ([XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)), with its Pi_05 adapter and vendored openpi |
| Checkpoint | Hugging Face dataset `TianxingChen/RoboTwin2.0`, revision `3dc3b79`, file `cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz` (12,443,100,901 bytes, SHA256 `957b07d4319746c70ed5173c2b0ae11b96872410d139931de603d9b4c974ee1c`); step directory `29999`, normalization `assets/RoboTwin_lerobot_v30` |
| Simulation assets | `embodiments.zip`, `objects.zip`, `background_texture.zip` from the same dataset revision (no training demonstrations needed) |
| Leaderboard numbers | [RoboTwin 2.0 leaderboard](https://robotwin-platform.github.io/leaderboard) data of 2026-10-11, used only for step-count estimates ([subset](../evalopt/data/robotwin_leaderboard_pi05.json)) |

## Environment

Measured on 8× NVIDIA H20 (96 GB each, sm_90, no RT cores), 128 CPU cores, driver 535.183.06.

| Environment | Packages |
|---|---|
| Simulator | Python 3.10.20, torch 2.4.1+cu121, NumPy 1.26.4, SAPIEN 3.0.0b1, mplib 0.2.1, cuRobo 0.7.8 (built for sm_90 with the CUDA 12.6 toolkit), warp-lang 1.12.0; installed with upstream `scripts/_install.sh` (including its SAPIEN URDF and mplib fixes) |
| Policy | `XPolicyLab/policy/Pi_05/openpi/.venv` from `XPolicyLab/policy/Pi_05/install.sh`: Python 3.11.17, JAX 0.5.3 (CUDA 12), torch 2.10.0, lerobot 0.4.4 |
| System | `libvulkan1`, `vulkan-tools`, `ffmpeg`, `libegl1` |

Headless SAPIEN rendering needs the NVIDIA Vulkan ICD and EGL vendor files; if the host does not provide them,
write them yourself and export `VK_ICD_FILENAMES` and `__EGL_VENDOR_LIBRARY_FILENAMES`
(see [activate-sim.example.sh](../scripts/evalopt/activate-sim.example.sh)). The policy environment has an upstream
metadata conflict (optional `rerun-sdk` wants NumPy ≥ 2, openpi pins NumPy < 2) that does not affect serving.

The upstream shell launchers (`scripts/eval_policy.sh`, `scripts/eval_policy_multitask.py`) activate a conda
environment. The measured setup used virtual environments and called the official Python entry points directly
(`XPolicyLab/setup_policy_server.py`, `scripts/eval_policy_xpolicylab.py`), which is what the launchers here do.
`run_plain.sh` uses `scripts/eval_policy.sh` and needs that activation adapted.

## Official protocol (unchanged)

| Item | Setting |
|---|---|
| Checkpoint | released co-train checkpoint, explicit `29999` directory; `repo_id: RoboTwin_lerobot_v30` (the generic `deploy.yml` defaults, 59999 and `arx_x5_sim`, do not match the release) |
| Tasks | the 50 tasks of `env_cfg/eval/all_tasks.yml` |
| Episodes | 100 accepted episodes per task and setting (10,000 in total) |
| Settings | `demo_clean` (seen instructions) and `demo_randomized` (unseen instructions, original domain randomization) |
| Expert check | `expert_check=true`; unstable or failed expert seeds are skipped by the official loop |
| Seeds | CLI seed 0, raw seeds from 100000, official progression |
| Step limit | per task, `_eval_step_limit.yml` (400–1700) |
| Observation / action | head + two wrist RGB cameras, 14-D joint state; joint actions, native 50-step action chunks |
| Sampling | JAX default, 10 flow steps |
| Output | per-task success rate and one eval video per episode (head camera) |
| Official schedule | `eval_policy_multitask.py`: 8 GPUs, 1 job (task × setting) per GPU, 1 simulator worker per job, a new policy server per job on the same GPU |

[`official-pi05-deploy.yml`](../scripts/evalopt/configs/official-pi05-deploy.yml) is the upstream `deploy.yml` with
the released step and normalization selected; `serve_policy.sh` adds the port and the checkpoint path.

## Launch

```bash
# once: paths (see scripts/evalopt/env.sh)
cp scripts/evalopt/activate-sim.example.sh scripts/evalopt/activate-sim.sh   # then edit
export PI05_CKPT=/path/to/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999

source scripts/evalopt/configs/faster.env      # the measured switches
mkdir -p runs records
# first evaluation of a job: run the official expert check and record its outcomes
python evalopt/make_jobs.py runs/jobs_w1.json 19001 0 runs/gpu0/w1 \
  place_container_plate:demo_clean:100:record=records/place_container_plate_clean.pkl
# later evaluations (any checkpoint) use ...:replay=records/place_container_plate_clean.pkl instead
# one multi-session server + one resident worker per jobs.json, all on GPU 0, port 19001
bash scripts/evalopt/run_ms_workers.sh 0 19001 runs/gpu0 runs/jobs_w1.json   # add more jobs.json for more workers
```

`make_jobs.py` writes the official evaluator arguments for each job (`--seed 0 --expert_check true --eval_batch false`
and the given `--test_num`). Each job writes `episodes.jsonl` (per-episode fingerprints), `events.jsonl`,
`stats.json` and the official `eval_result/` tree with its videos. Expert-check records do not depend on the policy
and can be reused for every checkpoint; the throughput numbers in this repository were measured with replayed
records, and a recording run was not timed at scale.

The launchers are the measured scripts with the workspace paths replaced by the variables of
`scripts/evalopt/env.sh`. After that change they were syntax-checked and checked statically against the pinned
upstream commits (every patched function and attribute exists; the two copies differ from upstream only in their
marked lines), but not re-run; run one short `run_pair.sh` case on a new machine first.

Before trusting a run, check every `eval.log` / `worker.log` for `Policy rollout error`, `env error before first
action` and `Failed to generate episode instruction`, and check that each job has `--test_num` episodes
([validation](validation.md#valid-runs)).

**Not included:** a scheduler for all 100 jobs. The measured deployment shape is one `run_ms_workers.sh` per GPU
with one `jobs.json` per worker. A full run additionally needs a global longest-first queue, staggered worker start,
a per-worker exit timeout and automatic reruns of invalid jobs ([known issues](#known-issues)); this was designed
from the measurements but not built or run.

## Switches

| Switch | Side | What it does | Files | Round | Status |
|---|---|---|---|---|---|
| `EVALOPT_SKIP_UNUSED_CAMS=1` | sim | Never render the front camera: the policy client reads head/left/right and the video writes head only. | `prof_eval.py` | R02 | adopted, E0 |
| `EVALOPT_LAZY_OBS=1` | sim | Inside an action chunk, render only the head camera (eval video) and skip `update_obs`; at every chunk start the full `get_obs()` + `update_obs` + `get_action` run as before. `_update_render()` is never skipped. Valid because the π0.5 server keeps only the latest observation. | `opt_loop.py`, `prof_eval.py` | R02 | adopted, E0 |
| `EVALOPT_EXPERT_RECORD=<pkl>` | sim | Run the expert check for real and store, per seed, how it ended and the `play_once` info. | `expert_cache.py` | R04 | adopted |
| `EVALOPT_EXPERT_REPLAY=<pkl>` | sim | Replay the stored outcomes: no simulation in the expert phase; the eval phase runs for real. | `expert_cache.py` | R04 | adopted, E0 against a reference replaying the same record |
| `prof_eval.py --jobs jobs.json` | sim | One resident simulator process runs many jobs (import, `Sapien_TEST`, renderer start-up paid once). | `prof_eval.py`, `run_resident.sh` | R07 | adopted, E0 |
| `EVALOPT_SERVER_NEW_JOB=1` | sim → server | At job start, ask a reused server to restart its JAX random-number chain at `key(0)` and drop its observation (new RPC `evalopt_new_job`). Set by `run_resident.sh`. | `prof_eval.py`, `prof_server.py` | R07 | adopted, E0 |
| `EVALOPT_LAZY_CUROBO=1` | sim | Build each cuRobo planner on first use; `plan_grippers` (a pure function) runs without building it. With replayed expert checks the planner is never built. | `prof_eval.py` | R07 | adopted, E0 |
| `EVALOPT_MEM_FRACTION=0.08` | server | JAX allocator pool share instead of the official 0.3. | `serve_policy.sh` | R03, R06 | adopted, E0 |
| `EVALOPT_MULTISESSION=1` | server | One server process per GPU, one session per `evaluation_id`: shared parameters and jitted functions, per-session observation and `key(0)` random-number chain; model calls stay serialized by the official lock. Set by `run_ms_workers.sh`. | `multisession_server.py` | R11 | adopted, E0 |
| `EVALOPT_CUDA_BLOCKING_SYNC=1` | sim | Primary CUDA contexts use blocking sync, so waiting for the GPU no longer spins a CPU core (76.9 → 0.8 ms CPU per frame). | `cuda_sched.py` | R10, R11 | adopted, E0 |
| `EVALOPT_KEEP_CACHE=1` | sim | Never call SAPIEN `clear_cache` (official: every 5 episodes; the next first frame then costs 13–40 s of CPU). | `prof_eval.py` | R10, R12 | adopted, E0 |
| `EVALOPT_AUTOTUNE=<file>` | server | XLA autotune results loaded from a file recorded on an idle GPU (`--xla_gpu_load_autotune_results_from`); default `evalopt/autotune/pi05_h20_idle.textproto`, empty string disables. Applies to reference and candidate alike. | `serve_policy.sh` | R01 | always on in measurements |
| `EVALOPT_NO_SUBSTEP_RENDER=1` | sim | `take_action` without the render sync after each physics substep (kept when crazy random light is on). | `opt_take_action.py` | R08, R10 | E0, **not adopted** (no throughput gain when the GPU is saturated) |
| `EVALOPT_NO_VIDEO=1` | sim | Turn the task's `eval_video_log` off: no head frame per step during evaluation. | `prof_eval.py` | R09, R17 | **optional, not default**: changes when videos are produced |
| `EVALOPT_RECORD_ACTIONS=1` | sim | Save every episode's action chunks to `<out>/actions/<seed>.pkl` for `replay_video.py`. | `prof_eval.py` | R09 | optional |
| `EVALOPT_FRAME_LOG=<file>` | sim | Log every camera picture (camera, seed, ms). | `prof_eval.py` | R19 | measurement only |
| `EVALOPT_DUMMY_POLICY=1` | sim | No policy server; each episode ends after one no-op action. | `prof_eval.py` | R04 | diagnosis only |

`scripts/evalopt/configs/faster.env` sets the adopted switches of the measured configuration.
`scripts/evalopt/configs/deferred-video.env` adds the optional deferred-video switches.

### Deferred video (optional)

With `EVALOPT_NO_VIDEO=1 EVALOPT_RECORD_ACTIONS=1`, the evaluation renders only the three policy cameras at each
chunk start. Afterwards,

```bash
python evalopt/replay_video.py --task place_container_plate --config demo_clean \
  --actions <job out>/actions --ref <job out> --out replay.json --mp4_dir <video dir>
```

rebuilds the eval scene with the evaluator's own `setup_demo(seed)`, replays the recorded chunks with the official
loop's control flow and writes `episode<k>.mp4` with the evaluator's ffmpeg command. Rebuilding every video costs
about as much GPU time as rendering it during evaluation (20–105 s per episode); the gain is that scores arrive first
and videos can be rebuilt for selected episodes. Evidence and limits are in [validation](validation.md#deferred-video).

## Deployment shape that was measured

- One multi-session policy server per GPU (`EVALOPT_MULTISESSION=1`, memory share 0.08) and K resident simulator
  workers on the same GPU, K = 10 or 14. Ten sessions used 44 GB; 14 jobs per GPU peaked at 58–69 GB with deferred
  video.
- Each worker runs its jobs back to back (`prof_eval.py --jobs`), with replayed expert checks, lazy cuRobo, blocking
  CUDA sync and the render cache kept.
- All servers load the same pinned autotune file. A resident server must serve one checkpoint; restart it to change
  checkpoints.

## Known issues

- **Renderer errors are silently counted as failures by the official loop.** Under high host load (about 65 render
  processes host-wide) one episode hit `vk::Device::createFenceUnique: ErrorOutOfHostMemory` at step 376 of 800; the
  official loop records an exception during the rollout as a failed episode. Treat any `Policy rollout error` as an
  invalid job and rerun it. A 12-episode resident-worker run showed no growth of file descriptors, memory maps,
  threads or RSS, so this is not a leak of the resident process.
- **Zombie workers.** After that error a worker finished its job list, but the Python process stayed a zombie, its
  launcher blocked in `wait`, and the multi-session server on that GPU did not exit (holding 8 GB) until it was
  stopped by hand. A scheduler needs an exit timeout per worker and must kill on timeout.
- **Staggered start.** Starting 70 render processes on 8 GPUs at once made each frame about 5× slower (451 ms instead
  of 78 ms, GPU 4% busy). Start workers one after another.
- **Ports on shared hosts.** Another service may already listen on a chosen local port; a client then talks to the
  wrong server (`msgpack decode failed`, `timeout waiting for hello_ack`, endless retries). The launchers refuse busy
  ports and only accept a listener owned by the current user (`ss -ltnp` shows `pid=`).
- **Do not edit a running shell script in place.** bash reads scripts by byte offset while running; replacing the
  file contents under a running launcher made it re-run a finished reference arm. Write a new file and `mv` it.
- **Instruction fallback.** Upstream `build_instruction` falls back to the task name on errors; a run with
  `Failed to generate episode instruction` is invalid.
- **Do not use `num_workers > 1` with π0.5.** In the official batch path several simulator workers share one server
  and one stateful model; `update_obs` and `get_action` take the lock separately, so one worker can act on another
  worker's observation.
- **Wrappers must forward arguments.** `open_laptop` and `open_microwave` call `check_success(target=...)` inside
  `play_once`; an early version of the record wrapper rejected the keyword and silently filtered every seed.
  `expert_cache.py` forwards `*args, **kwargs` and only intercepts the loop's own argument-free call.
