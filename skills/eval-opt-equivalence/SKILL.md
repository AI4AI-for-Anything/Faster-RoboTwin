---
name: eval-opt-equivalence
description: What "no numerical change" means for closed-loop evaluation and how to judge it: tiers E0–E3, making the reference reproducible first (pin XLA autotune), per-episode fingerprints, the two random-number chains, cross-talk risk with stateful policies, render statelessness checks. Read before changing the evaluation flow or the schedule, or skipping any computation.
---

> Adapted from `train-opt-numerics-validation` in [ResearchBoost-Agent](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/main/skills) for closed-loop policy evaluation. Measured on RoboTwin `ea8b211` + the official π0.5 co-train checkpoint on 8× H20; re-verify on your setup. Findings checked on RoboTwin + official π0.5 are stated as conclusions; unchecked ones are marked "to be measured".

# Equivalence for evaluation optimization

Training optimization asks whether a trajectory stays inside the reference's drift band. Evaluation optimization is stricter: **an evaluation measures the policy, so the evaluation pipeline itself may not have any numerical freedom.** The default tier here is bitwise.

## Tiers E0–E3

| Tier | Definition | Examples | How to judge | Who decides |
|---|---|---|---|---|
| E0 bitwise | Every episode's fingerprint (below) is bitwise identical to the reference | Skipping renders nobody reads; a resident policy server whose random-number chain is reset per task; pinned autotune / compile caches; task-level concurrency; caching policy-independent expert-check results | `evalopt/compare_arms.py`: every field of every episode equal | Automatic |
| E1 same distribution | Every inference computes bitwise the same numbers, but the order in which random numbers are drawn changes; single episodes differ and the success rate is another sample from the same distribution | Splitting one task's 100 episodes across several processes (the policy's JAX random-number chain and cuRobo's in-process state both change) | Cannot be judged bitwise; report it only as "another sample under the same protocol", together with the reference's own run-to-run spread | User |
| E2 numerical change | The numbers of inference or rendering change | Cross-environment batched inference (GEMM shape changes → kernel changes), rendering on another GPU architecture, changing spp / denoiser, changing flow steps, bf16→fp8 | Not done | Not done |
| E3 protocol change | Changes what is measured | Fewer episodes, expert_check off, different seed start, different step limit, different instructions, no video | Not done (no video is a separate user decision: it does not affect the success rate but drops an official artifact) | User |

The tier is set by measurement, not by declaration: if you changed the schedule or skipped a computation, run a per-episode comparison. If it does not match, it is not E0, however "equivalent in theory" it is.

## Step one: make the reference reproducible

A bitwise judgement requires the reference to be bitwise identical to itself across two runs. **The official π0.5 server cannot do that:**

- On the first inference JAX/XLA autotunes GEMMs/convolutions and picks a Triton tile or cuDNN algorithm by measured time. When the machine is busy (other processes on the same GPU or host), the choice changes, the rounding order changes with it, the actions change, and so does the rest of the trajectory.
- Measured (H20, same observation, same noise): on an idle GPU two processes differed in 3 of 25 autotune entries but produced bitwise-identical outputs; under 2 render processes + 48 CPU busy loops one run was still bitwise identical, while another differed in 8 choices and **its action output changed**. Also seen on whole episodes: same seed, same configuration, one of three runs took 125/114 steps instead of 123/118.
- The official 50-task scheduler starts 8 servers and 8 simulators on 8 GPUs at once, so all autotuning happens under load. **The official results are therefore not bitwise reproducible from run to run** (they are statistically equivalent).

Fix (E0; no precision change, it only fixes one of the reference's own legal choices): record the autotune results once on an idle GPU, then load them in every server with `XLA_FLAGS=--xla_gpu_load_autotune_results_from=<file>`. Reference and candidate arms must use the same file. Re-record after changing GPU model, driver or JAX version.

## Per-episode fingerprint

`evalopt/prof_eval.py` records per episode, and `evalopt/compare_arms.py` compares all of:

- seed, instruction text, success, step count, number of get_action calls;
- the hash of every observation the policy **actually consumed** (three RGB cameras + all state + instruction);
- the hash of every action chunk the policy returned;
- the hash of every frame written into the eval video, and the frame count.

Comparing only success rates is not enough: two runs with the same success rate may already have different intermediate actions.

## Step two: two random-number chains + the expert check (think this through before parallelizing)

1. **The policy server's JAX random numbers.** `Policy._rng` starts at `key(0)` and is split once per `infer`. The flow noise of episode k depends on how many inferences this process has run so far, that is, on how many steps all earlier episodes took. Officially every task starts a new server, so the chain is reset per task.
   - When a resident server is reused across tasks, `_rng` must be reset to `key(0)` at the start of each task to be equivalent to the official flow (E0).
   - Splitting one task's episodes across processes breaks the chain: E1.
2. **The expert check (cuRobo planning + physics execution + check_success).** Every `setup_demo` calls `np.random.seed(seed)` + `torch.manual_seed(seed)`, so scene generation is reproducible per seed; the expert check, however, has two kinds of nondeterminism:
   - In-process history: cuRobo's Halton sampler keeps advancing across episodes, so running the expert demonstration for the same seed a second time in the same process plans a different path.
   - **Not reproducible across processes:** for seed 100001 of place_dual_shoes / randomized, 4 identical processes (same inputs, same call order, policy replaced by a dummy client that ends the episode immediately) accepted it twice and rejected it twice (planning succeeded every time; after execution `check_success` was true in some and false in others). So **"which 100 seeds get evaluated" under the official protocol can differ from run to run**, independent of any optimization.
   - Consequence: two official runs may evaluate different seeds after some episode, and from then on the whole task no longer matches (the policy's random-number chain is shifted too). Remove this before judging bitwise: record one realization with `evalopt/expert_cache.py` and replay the same record in both the reference and the candidate arm. Replay replaces only the expert phase; the evaluation phase runs as usual. The record is one legal realization of the official filtering and has the same distribution.
   - Measured: the evaluation episode itself (setup_demo → inference/execution) is not affected by earlier expert demonstrations. With or without an expert demonstration before it, the first evaluation frame and the observation after 30 steps are bitwise identical.

## Stateful policies and concurrency

- `Pi_05.Model` stores only "the observation of the most recent update_obs" (`observation_window`, overwritten, no history). So an update_obs in the middle of an action chunk has no effect on the next get_action; this is the premise that makes "skip mid-chunk update_obs" E0. Re-check it for any other policy; it does not hold for policies with an observation history window.
- The official ws server has one model object per process, and `update_obs` and `get_action` each take the lock separately. **Several clients connected to one server (the official `num_workers>1` batch path) cross-talk:** A updates, B updates, and A's get_action uses B's observation. The official comments also say the batch path is only for stateless policies. For π0.5 this path is not "faster" but "measuring the wrong thing".
- The correct way to share weights across sessions: one independent policy wrapper per connection inside one process (its own observation buffer, its own random-number chain starting at `key(0)`), sharing one copy of the parameters and one compiled function (`evalopt/multisession_server.py`).

## Render statelessness (prove this before skipping renders)

Skipping the render at some steps is E0 if and only if rendering does not change physics state, does not consume random numbers used elsewhere, and a frame's pixels depend only on the scene at that moment and not on how often it was rendered before. Measured (RoboTwin rt shader, 32 spp, OIDN denoiser, H20):

- Same scene, same camera, calling `_update_render()` before every render: repeated renders are bitwise identical. **Taking a picture again without calling `_update_render()` changes the pixels.** So when skipping renders, not a single `_update_render()` may be dropped (with random lighting it also draws from `np.random`).
- "Render every camera at every step" and "render only at the last step" give a bitwise-identical last frame and a bitwise-identical physics state sequence; the same holds when the scene stays still for 25 steps.
- 1–8 processes rendering concurrently on the same GPU produce bitwise-identical pixels.

## Common pitfalls

- A dry run does not load the model and is not evidence of correctness.
- The official `build_instruction` silently falls back to task_name as the instruction on errors. Grep every run for `Failed to generate episode instruction`; a hit invalidates the run.
- Count and compare every skipped seed (unstable, expert failure, expert error, env error before the first action); the skip sequences of reference and candidate must match.
- Compare the "run fingerprint" before the episode fingerprints: command line, configuration, checkpoint path, autotune file, code revision. If they differ, refuse to compare.
