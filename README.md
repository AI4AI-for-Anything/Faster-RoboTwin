# Faster-RoboTwin

[English](README.md) | [简体中文](README.zh-CN.md)

**About 4.4× faster official π0.5 evaluation on RoboTwin 2.0 (estimated for the full benchmark on 8× H20), with every compared episode bitwise identical to the official pipeline, optimized by AI agents.**

![Faster-RoboTwin: estimated 4.38× wall-clock speedup of the full official π0.5 evaluation on 8× H20](docs/performance-views/01-estimate.svg)

| Pipeline | Wall clock on 8× H20, estimated ↓ | GPU time, estimated ↓ | Speedup ↑ |
|---|---:|---:|---:|
| Official (`eval_policy_multitask.py`, 1 job per GPU) | 73.5 h | 538.8 GPU-h | 1.00× |
| **Faster-RoboTwin** | **16.8 h** | **127.9 GPU-h** | **4.38×** (GPU time 4.21×) |

The workload is the official RoboTwin 2.0 evaluation of the official π0.5 co-train checkpoint (step 29999): 50 tasks × {clean, randomized} × 100 accepted episodes, expert check on, native 50-step action chunks, 10 flow steps, and an eval video for every episode. **These are estimates, not a measured full run.** [`evalopt/full_estimate.py`](evalopt/full_estimate.py) combines per-step and per-episode costs measured on 6 of the 100 jobs (8% of the steps; official costs from 2–10-episode runs, Faster-RoboTwin at 10–14 jobs per GPU) with step counts derived from the public leaderboard success rates (3.85 M steps). Jobs that were not measured use the mean of the measured jobs of their setting. The Faster column replays recorded expert-check outcomes, which are policy-independent and recorded once per job. Neither pipeline was run over all 10,000 episodes. See [measurement and estimate](docs/measurement.md).

## Quick Start

Install RoboTwin, its simulator environment and the Pi_05 policy environment as upstream describes ([setup and versions](docs/evaluation.md)), download the official checkpoint, then:

```bash
git clone --recurse-submodules https://github.com/AI4AI-for-Anything/Faster-RoboTwin.git
cd Faster-RoboTwin
cp scripts/evalopt/activate-sim.example.sh scripts/evalopt/activate-sim.sh   # edit: simulator env + Vulkan/EGL
export PI05_CKPT=/path/to/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999

source scripts/evalopt/configs/faster.env     # the measured switches
mkdir -p runs records
python evalopt/make_jobs.py runs/jobs_w1.json 19001 0 runs/gpu0/w1 \
  place_container_plate:demo_clean:100:record=records/place_container_plate_clean.pkl
bash scripts/evalopt/run_ms_workers.sh 0 19001 runs/gpu0 runs/jobs_w1.json   # one jobs.json per worker
```

`run_ms_workers.sh` starts one multi-session policy server per GPU and one resident simulator worker per `jobs.json`; the measured runs used 10–14 workers per GPU. The first run of a job records its expert-check outcomes (`record=`); later runs, also of other checkpoints, replay them (`replay=`). Measured with the official environments (simulator: Python 3.10, SAPIEN 3.0.0b1, cuRobo 0.7.8; policy: JAX 0.5.3) on **8× H20**. With no switch set, the wrappers run the official code. The scripts here are the measured ones with workspace paths replaced by variables (`scripts/evalopt/env.sh`); after that change they were syntax-checked and statically checked against the pinned upstream commits, not re-run. A scheduler for all 100 jobs (global longest-first queue, staggered start, worker timeouts, reruns) is not included; see [launch](docs/evaluation.md#launch).

## Agent Optimization History

Rounds R00–R19, run on 2026-10-17/18. Each bar is a measured configuration on one task whose episodes were bitwise identical to the official reference; the per-task ratios below it are estimates for 100-episode jobs.

![Measured throughput ladder on place_container_plate and per-task estimates](docs/performance-views/02-ladder.svg)

See the [round-by-round notes](docs/rounds.md).

## What changed

Every switch is an `EVALOPT_*` environment variable, off by default. Upstream RoboTwin and XPolicyLab files are not modified: the switches live in wrappers under [`evalopt/`](evalopt) and in two minimal-difference copies of upstream functions (`opt_loop.py`, `opt_take_action.py`, one marked change each).

- **Render only what is read:** the front camera, which neither the policy nor the video reads, is not rendered; inside an action chunk only the head camera is rendered for the video, and `update_obs` is skipped. The policy still receives the full head and wrist observation at every chunk start.
- **Expert checks recorded once:** the per-seed outcome of the official expert check is recorded once and replayed, which removes the cuRobo planning and physics from every later run and makes whole tasks comparable bit for bit.
- **Resident processes:** one simulator process and one policy server run many jobs; the server is reset to `key(0)` at each job, and cuRobo is built only when used. Fixed cost per job drops from 45–60 s to 2–3 s.
- **Many jobs per GPU:** one multi-session policy server per GPU (shared weights, per-job observation and random-number chain, JAX memory share 0.08) and 10–14 resident workers per GPU.
- **Host-side waits and caches:** CUDA waits block instead of spinning (CPU per frame 76.9 → 0.8 ms), and SAPIEN's render cache is no longer cleared every 5 episodes.
- **Reproducible reference:** XLA autotune results are pinned to a file recorded on an idle GPU, for the reference and the candidate alike.

![Optimization map](docs/change-views/module-map.svg)

![Where the time goes, official vs. Faster-RoboTwin](docs/performance-views/03-time-budget.svg)

See the [switch table](docs/evaluation.md#switches) and the [measured configuration](scripts/evalopt/configs/faster.env).

## Validation

Every adopted switch is **E0**: per episode, the candidate matches the official reference bit for bit in the seed, the instruction, success, the number of steps and of policy queries, the hash of every observation the policy consumed, the hash of every action chunk it returned, and the hash of every frame written to the eval video ([`compare_arms.py`](evalopt/compare_arms.py)). The reference is the official functions with timing and hashing wrappers; the same case run through the plain official entry gave eval videos with identical md5.

- Seven tasks were compared (adjust_bottle, blocks_ranking_rgb, place_container_plate, stack_blocks_three clean; handover_block, open_laptop, place_dual_shoes randomized), including failed episodes that ran to 800 and 1200 steps. In the deployment shape (multi-session server, 6–14 jobs per GPU) 136 jobs matched; one more job hit a renderer error under host-wide load, which the official loop counts as a failure, and is treated as invalid.
- All comparisons are smoke-scale (2–10 episodes per job). The two longest tasks (step limits 1500–1700) were not compared, and no 100-episode job was compared end to end.
- The official pipeline is not bitwise reproducible by itself, so three sources were fixed or controlled first: XLA autotune choices change under load (pinned to a file), the expert check accepts or rejects the same seed differently across processes (one record, replayed by both arms), and instructions depend on `--test_num` (smoke runs are not subsets of the formal run).

See the [validation method](docs/validation.md).

## Optional: deferred video

Not the default and not part of the headline. With `EVALOPT_NO_VIDEO=1 EVALOPT_RECORD_ACTIONS=1` the evaluation skips the head-camera frame on every step and saves the action chunks; [`replay_video.py`](evalopt/replay_video.py) rebuilds the videos afterwards. Scores and all other fingerprints matched the official pipeline in 84 jobs at 14 jobs per GPU and 25 jobs at 1–12 jobs per GPU; rebuilt mp4 files were md5-identical to the official ones on **6 episodes** (2 tasks). Time to scores is estimated at about 13–18× (4.1–5.7 h on 8 GPUs). Rebuilding every video costs about as much GPU time as rendering it during evaluation; the mode changes when videos are produced and needs an explicit decision ([details](docs/evaluation.md#deferred-video-optional)).

## Why it stopped near 4.4×

With the eval video written during evaluation, each step needs one ray-traced head-camera frame (32 samples per pixel, about 92 ms on H20, which has no RT cores), and frames from different processes or threads do not overlap on the GPU. On place_container_plate the final configuration reached 96% of the per-step GPU demand; a 5× pace on that task would need a step shorter than the head frame alone. The estimated ceiling for the whole benchmark is about 4.6×.

![Why it stops near 4.4×](docs/performance-views/04-ceiling.svg)

See [why it stopped here](docs/measurement.md#why-it-stopped-near-44) and the directions that were tried and not adopted.

## More AI4AI-for-Anything projects

[Faster-JiT](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## Upstream RoboTwin documentation

Based on [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) at `ea8b211` (XPolicyLab submodule `fa431ec`), by the RoboTwin team (Tianxing Chen, Yao Mu et al.). [Paper](https://arxiv.org/abs/2506.18088) · [Documentation](https://robotwin-platform.github.io/doc/) · [Leaderboard](https://robotwin-platform.github.io/leaderboard) · [MIT license](LICENSE). The official π0.5 checkpoint and simulation assets come from the [RoboTwin 2.0 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0). The original README is kept in [README.upstream.md](README.upstream.md) and below.

<details>
<summary>Original RoboTwin README: installation, data, usage and citation</summary>

<h1 align="center">
  <a href="https://robotwin-benchmark.github.io"><b>RoboTwin</b> Bimanual Robotic Manipulation Platform<br></a>
</h1>
<h2 align="center">Latest Version: RoboTwin 2.0<br>🤲 <a href="https://robotwin-platform.github.io/">Webpage</a> | <a href="https://robotwin-platform.github.io/doc/">Document</a> | <a href="https://arxiv.org/abs/2506.18088">Paper</a> | <a href="https://robotwin-platform.github.io/doc/community/index.html">Community</a> | <a href="https://robotwin-platform.github.io/leaderboard">Leaderboard</a></h2>

https://private-user-images.githubusercontent.com/88101805/463126988-e3ba1575-4411-4a36-ad65-f0b2f49890c3.mp4

**RoboTwin 2.0** (*ICML 2026*) — [Webpage](https://robotwin-platform.github.io/) · [Doc](https://robotwin-platform.github.io/doc) · [Paper](https://arxiv.org/abs/2506.18088) · [Talk](https://www.bilibili.com/video/BV18p3izYE63/?spm_id_from=333.337.search-card.all.click) · [机器之心](https://mp.weixin.qq.com/s/SwORezmol2Qd9YdrGYchEA) · [Leaderboard](https://robotwin-platform.github.io/leaderboard)

<details>
<summary>Earlier papers & challenge report</summary>

- **1.0 / Early** — RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins · *CVPR 2025 (Highlight)* [PDF](https://arxiv.org/pdf/2504.13059) / [arXiv](https://arxiv.org/abs/2504.13059) · *ECCV Workshop 2024 (Best Paper)* [PDF](https://arxiv.org/pdf/2409.02920) / [arXiv](https://arxiv.org/abs/2409.02920)
- **CVPR'25 MEIS Challenge Report** — [PDF](https://arxiv.org/pdf/2506.23351) / [arXiv](https://arxiv.org/abs/2506.23351) · [量子位](https://mp.weixin.qq.com/s/qxqs9vvvHsAJ-0hoYANYzQ)

</details>

# 📚 Overview

> Prefer the [RoboTwin Document](https://robotwin-platform.github.io/doc/) for full guides — this README is a quick start.

RoboTwin 2.0 and [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) share deployment via [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab): one policy-serving / eval stack across both benchmarks (single-task, multi-GPU, and remote server + local sim).

**Default branch:** [`main`](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (RoboTwin 2.0).

<details>
<summary>Other branches (legacy / special-purpose)</summary>

- [IsaacLab-Arena](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena) · [RLinf_support](https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support) · [WBCD-2026](https://github.com/RoboTwin-Platform/RoboTwin/tree/WBCD-2026)
- [RoboTwin-1.0](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0) / [early_version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version) · [gpt](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
- [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) · [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1) / [Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

</details>

# 🐣 Update
* **2026/08/03**, We add [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)-based policy evaluation with single-task evaluation, multi-task multi-GPU scheduling, and remote policy-server/local-simulator deployment.
* **2026/03/03**, We release [RMBench](https://github.com/RoboTwin-Platform/RMBench), which is a memory-dependent manipulation benchmark built upon RoboTwin 2.0.
* **2026/02/20**, Usage supported in <a href="https://github.com/starVLA/starVLA">StarVLA</a>, which is a user-friendly codebase for VLA development.
* **2026/01/23**, We update IsaacLab-Arena and <a href="https://github.com/RLinf/RLinf">RLinf</a> support (contributed by RLinf team).
* **2025/08/28**, We update the RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088).
* **2025/08/25**, We fix ACT deployment code and update the [leaderboard](https://robotwin-platform.github.io/leaderboard).
* **2025/08/06**, We release RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
* **2025/07/23**, RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
* **2025/07/19**, We Fix DP3 evaluation code error. We will update RoboTwin 2.0 paper next week.
* **2025/07/09**, We update endpose control mode, please see [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for more details.
* **2025/07/08**, We upload [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支).
* **2025/07/02**, Fix Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
* **2025/07/01**, We release Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
* **2025/06/21**, We release RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
* **2025/04/11**, RoboTwin is selected as <i>CVPR Highlight paper</i>!
* **2025/02/27**, RoboTwin is accepted to <i>CVPR 2025</i> ! 
* **2024/09/30**, RoboTwin (Early Version) received <i>the Best Paper Award  at the ECCV Workshop</i>!
* **2024/09/20**, Officially released RoboTwin.

# 🛠️ Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for installation instructions. It takes about 20 minutes for installation.

XPolicyLab is embedded as a Git submodule. For a fresh checkout, clone RoboTwin recursively:

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
```

For an existing checkout, initialize the version pinned by RoboTwin:

```bash
git submodule update --init --recursive XPolicyLab
```

To pull the latest XPolicyLab commit on its configured `main` branch and refresh RoboTwin's submodule pin:

```bash
bash scripts/update_xpolicylab.sh
# optional: stage the pin / reinstall the editable package
bash scripts/update_xpolicylab.sh --stage --install
```

# 🤷‍♂️ Task Information
See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

# 🧑🏻‍💻 Usage 

## Document

> Full usage details live in the [RoboTwin Document](https://robotwin-platform.github.io/doc/) — start from [Usage](https://robotwin-platform.github.io/doc/usage/index.html). Prefer the Doc over this README when anything conflicts.

## Getting Data
We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset). **We recommend downloading the pre-collected data (step 1) as the default path** — it is ready to train on immediately. Collect data yourself (step 2) only when you need custom task configs, domain randomization, or embodiment setups.

> **Always decode through `decode_image_bit`, and always encode through `encode_image_bit`.** Downloaded and self-collected episodes store cameras as encoded image bits. Decoding them yourself is unsupported, because they come in two byte formats and a hand-rolled decoder is right on one and reverses the channels on the other. A local copy of the pair is in [`data/decode_image_bit.py`](data/decode_image_bit.py) — it is the same functions as `XPolicyLab.utils.process_data`. Prefer the XPolicyLab import when the package is available:
>
> ```python
> from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit
> rgb = decode_image_bit(image_bits)  # RGB for every data version
> ```
>
> Stored image bits come in two formats, and both decode to RGB:
>
> | Format | How it was written | What a standard decoder sees |
> | --- | --- | --- |
> | **legacy** | an RGB array handed straight to `cv2.imencode`, which reads its input as BGR | red and blue swapped — the bytes are channel-reversed against the JPEG standard, and `cv2.imdecode` reverses them back |
> | **standard** | `encode_image_bit`, which converts to BGR first and stamps a JPEG `COM` segment with the payload `XPL-RGB1` | correct colors |
>
> Legacy data is never migrated — JPEG cannot swap channels losslessly — so the two formats coexist indefinitely and may appear in the same training run. Self-collected episodes now write the standard format. `decode_image_bit` tells the formats apart and returns RGB, so its output never needs a channel swap. Do **not** add `cv2.cvtColor(..., COLOR_BGR2RGB)` after it — the two formats are indistinguishable from any single sample, and that swap is the bug. Official LeRobot converters already call it; any custom training dataloader that reads these HDF5 files must do the same. Collection must not replace `encode_image_bit` with a bare `cv2.imencode`, which omits the marker and writes a buffer that reads back reversed. At eval time the policy server hands over decoded RGB, so `model.py` must not decode again. Details: [XPolicyLab — Decode only through `decode_image_bit`](https://github.com/XPolicyLab/XPolicyLab#decode-only-through-decode_image_bit).

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

## 1. Download XPolicyLab-Format Data (Recommended)
Download and extract all available XPolicyLab-format trajectories from Hugging Face:

```bash
bash scripts/download_xpolicylab_data.sh
```

To download only selected tasks, pass their names:

```bash
bash scripts/download_xpolicylab_data.sh adjust_bottle beat_block_hammer
```

Downloads and extractions both run in parallel (defaults: 8 workers each). As soon as a task ZIP finishes downloading, extraction starts without waiting for the rest. Tune concurrency with:

```bash
# Parallel download / extract workers (extract defaults to HF_MAX_WORKERS)
HF_MAX_WORKERS=8 HF_EXTRACT_WORKERS=16 bash scripts/download_xpolicylab_data.sh
```

Downloads land under `data/demo_clean/<task_name>/aloha_agilex/data/` (note: self-collected data lands under `data/<task_config>/...` instead).

## 2. Task Running and Data Collection (Optional)
For custom task configs, domain randomization, or embodiment setups, collect data yourself. The following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

Collected demonstrations are saved directly in the XPolicyLab trajectory format — no extra conversion step is needed:

```text
data/<task_config>/<task_name>/<embodiment>/data/episode_0000000.hdf5
```

`<embodiment>` follows the `embodiment` field of the task config (`aloha_agilex` for the default `aloha-agilex` setup).

## 3. Convert to LeRobot (Optional)

Many XPolicyLab policies train on LeRobot datasets. After you have XPolicyLab-format HDF5 under `data/<task_config>/<task>/<embodiment>/data/` (from download or collection), convert with the shared scripts in `XPolicyLab/scripts/`. Those scripts already decode through `decode_image_bit` — do not replace that with `cv2.imdecode` / PIL if you fork them. Stored bits come in two byte formats; only `decode_image_bit` tells them apart.

Patterns are `<task_config>.<task>.<embodiment>` and may use `*` wildcards. They resolve against `data/` next to the RoboTwin root (for example `demo_clean.*.aloha_agilex`). Keep `--data_type` as the default `RoboDojo` — RoboTwin XPolicyLab trajectories share that HDF5 layout.

Run in an environment that already has the matching LeRobot package (v2.1 vs v3.0). Conversion writes under `HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`); point it at a large disk if needed:

```bash
export HF_LEROBOT_HOME=/path/with/enough/space/lerobot

# LeRobot v2.1 — all demo_clean tasks
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.*.aloha_agilex" \
  --repo_id robotwin_demo_clean_aloha_agilex \
  --max_episode 50

# LeRobot v3.0 — same selection
python XPolicyLab/scripts/transform_lerobot_v30_format.py \
  "demo_clean.*.aloha_agilex" \
  --repo_id robotwin_demo_clean_aloha_agilex_v30 \
  --max_episode 50

# Single task
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.beat_block_hammer.aloha_agilex" \
  --repo_id beat_block_hammer_demo_clean
```

Useful flags: `--repo_id` (output dataset name), `--max_episode` (cap episodes per task/embodiment), `--resolution HxW` or `--image_height` / `--image_width` (default: auto-detect; RoboTwin is often `240x320`). Output lands at `${HF_LEROBOT_HOME}/<repo_id>`.

## 4. Modify Task Config
☝️ See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

Task settings such as `demo_clean` and `demo_randomized` are stored in `env_cfg/task_config/`.

## 5. Evaluate Policies via XPolicyLab

All evaluation goes through `scripts/eval_policy.sh`. The policy adapter must exist under `XPolicyLab/policy/<policy_name>/` (see the [XPolicyLab policy catalog](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy)).

`--env-cfg-type` selects the XPolicyLab action profile (validated against `XPolicyLab/utils/robot/_robot_info.json`; `arx_x5` matches RoboTwin's default aloha-agilex layout), while the simulator embodiment stays controlled by `--task-config`.

**Local evaluation (multi-task, multi-GPU).** The scheduler starts a policy server and simulator per task on your GPU pool. Task lists and GPU settings live in `env_cfg/eval/all_tasks.yml` (trim `tasks` to a single entry for single-task evaluation):

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name <policy_name> \
  --ckpt-name <checkpoint> \
  --env-cfg-type arx_x5 \
  --policy-conda-env <policy_env> \
  --eval-env-conda-env <robotwin_env> \
  --action-type <action_type>
```

Add `--dry-run` to validate the schedule without launching anything. Results are written to `eval_result/multitask/` by default.

**Split deployment (remote policy server + local simulator).** Start the server pool on the policy host, then point the local scheduler at it:

```bash
# On the policy-server host (fill in the placeholders first):
bash scripts/eval_policy.sh serve --config env_cfg/eval/remote_server.yml

# On the simulator host:
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name <policy_name> \
  --env-cfg-type arx_x5 \
  --eval-env-conda-env <robotwin_env> \
  --enable-remote \
  --policy-server-ip <server_ip> --policy-server-port <port>
```

`--policy-server-ip/--policy-server-port` can be repeated to use a server pool, or configured once via `enable_remote` / `policy_server_ip` / `policy_server_port` in the eval config.

# 🏄‍♂️ Experiment & LeaderBoard

> We recommend that the RoboTwin Platform can be used to explore the following topics: 
> 1. single - task fine - tuning capability
> 2. visual robustness
> 3. language diversity robustness (language condition)
> 4. multi-tasks capability
> 5. cross-embodiment performance

The full leaderboard and setting can be found in: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

# 💽 Pre-collected Large-scale Dataset

Please refer to [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

# 🎯 Official Benchmark Checkpoints

Please refer to [RoboTwin 2.0 Checkpoints - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/cotrain_ckpt).

# 👍 Citations
If you find our work useful, please consider citing:

<b>RoboTwin 2.0</b>: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
```
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

<b>RoboTwin</b>: Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>
```
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

<b>RoboTwin</b>: Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

# 😺 Acknowledgement

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

Contact [Tianxing Chen](https://tianxingchen.github.io) if you have any questions or suggestions.

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

</details>
