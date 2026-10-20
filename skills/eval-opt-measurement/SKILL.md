---
name: eval-opt-measurement
description: Pitfalls of timing and comparing closed-loop evaluations: valid-run criteria (instruction fallback, seed skips, rollout errors), alternating reference and candidate, time-sharing of rendering on one GPU, weighting by episode mix, full-benchmark time estimates, GPU ledger. Read before timing anything or comparing results.
---

> Adapted from `train-opt-measurement` in [ResearchBoost-Agent](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/main/skills) for closed-loop policy evaluation. Measured on RoboTwin `ea8b211` + the official π0.5 co-train checkpoint on 8× H20; re-verify on your setup.

# Timing and comparing evaluations

## Valid runs

Exit code 0 is not success. Check every run; any one of the following invalidates it:

- `Failed to generate episode instruction` (the instruction silently fell back to the task name);
- `Policy rollout error`, `worker_error`, `env error before first action`. **The official loop counts an exception in the middle of a rollout as a failed episode and includes it in the success rate as usual** (measured: under high concurrency one `vk::Device::createFenceUnique: ErrorOutOfHostMemory` caused an 800-step episode to be recorded as failed at step 376). So if it appears, mark the job invalid and rerun it; never accept it being silently counted;
- the number of episodes differs from `--test_num`;
- the seed-skip sequences (unstable / expert failure / expert error) of reference and candidate differ;
- the server log shows recompilation (the same shape compiled a second time), or autotuning after the first inference;
- the run fingerprint (command line, configuration file, checkpoint, autotune file, code revision, visible GPUs) differs from the reference.

## How to run reference and candidate

- Every arm **starts a new policy server** (random-number chain from `key(0)`), with server and simulator on the same GPU (the layout of the official `eval.sh`).
- Order: reference → candidate → reference (`scripts/evalopt/run_pair.sh` with `REPEAT_REF=1`). The two reference runs must also be bitwise identical per episode; otherwise fix reproducibility first.
- **Rendering is a serial resource of one GPU:** while timing, no other render process may run on the same GPU. Jobs on other GPUs of the same host compete for CPU; note them. On a shared machine, differences smaller than the host's fluctuation count as "not resolved".
- The per-episode comparison (`evalopt/compare_arms.py`) comes before timing: if the fingerprints do not match, do not look at speed.

## What to time

- Report separately: job start-up (once), per-episode fixed cost (expert check, scene setup) and per-step cost (rendering, physics, inference).
- Report two speed ratios: the rollout part (`t_start`→`t_end`, excluding the expert check) and the whole job's wall clock.
- Failed episodes run up to the step limit, so a task's total cost is dominated by its failed episodes. When extrapolating from a few episodes to the full benchmark, use "successful-episode steps × success rate + step limit × failure rate", and state where the success rate comes from (official leaderboard or own measurement).
- Smoke runs with few episodes (for example 2–3 per task) are only for correctness and localization and **are not results**; label them "smoke" in result directories and reports.

## Full-benchmark time estimate (official protocol, 2 settings × 50 tasks × 100 episodes)

Each job ≈ start-up + 100 × (expert check + 2 × scene setup) + expert cost of rejected seeds + Σ steps × per-step cost. The official default is 8 GPUs with 1 job per GPU, 100 jobs in total; the wall clock is set by the tail of the schedule (the longest jobs are the tasks with step limits of 1500–1700).

## GPU ledger

Record the start/end time and GPU count of every GPU job; respect the host's shared GPU lock if there is one, and release it when done.

## Long jobs

Start them with `setsid nohup` and wait in the session; when investigating, first check whether the artifacts exist, then look for the first traceback.

**Make sure the port is yours.** On a shared machine, other users' services on the same host also listen on 127.0.0.1 ports. If your server fails to bind, a "port is listening" readiness check is fooled by someone else's process and the client connects to their service; the symptom is `msgpack decode failed` + `timeout waiting for hello_ack`, retried forever. Require the port to be free before starting, and in the readiness check require `ss -ltnp` to show `pid=` (only shown for your own processes).

**Do not edit a running shell script in place.** bash reads a script file by byte offset while executing it; if the same inode is overwritten, a running instance continues reading the new file at the old offset and executes half a line or a different line. Measured: editing `run_pair.sh` while a comparison was running made it rerun the reference arm and overwrite 13 minutes of finished results. Write a new file and `mv` it over the old one (new inode), or wait until all instances have finished.
