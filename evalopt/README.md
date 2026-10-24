# evalopt

Wrappers, judges and measurement tools for the official RoboTwin π0.5 evaluation. Upstream files are not modified;
everything here imports the official modules from the repository root (override with `ROBOTWIN_ROOT`). Launchers are
in [`scripts/evalopt/`](../scripts/evalopt); switches and usage are in [docs/evaluation.md](../docs/evaluation.md).

| File | Role |
|---|---|
| `prof_eval.py` | Runs the official `scripts/eval_policy_xpolicylab.py` with timing wrappers, per-episode fingerprints and the simulator-side `EVALOPT_*` switches; `--jobs` runs many jobs in one resident process |
| `opt_loop.py` | Copy of upstream `eval_remote_policy` (`scripts/eval_policy_xpolicylab.py` lines 772–975) with one marked change (`EVALOPT_LAZY_OBS`) |
| `opt_take_action.py` | Copy of upstream `Base_Task.take_action` (`envs/_base_task.py` lines 1552–1740) with one marked change (`EVALOPT_NO_SUBSTEP_RENDER`, not adopted) |
| `expert_cache.py` | Expert-check record / replay (`EVALOPT_EXPERT_RECORD`, `EVALOPT_EXPERT_REPLAY`) |
| `cuda_sched.py` | `EVALOPT_CUDA_BLOCKING_SYNC`; imported before anything creates a CUDA context |
| `prof_server.py` | Official Pi_05 policy server with timing wrappers and the `evalopt_new_job` RPC |
| `multisession_server.py` | One server per GPU, one session per evaluation job (`EVALOPT_MULTISESSION`) |
| `replay_video.py` | Rebuilds eval videos from recorded action chunks (optional deferred-video mode) |
| `make_jobs.py` | Writes `jobs.json` for `prof_eval.py --jobs` (requires `PI05_CKPT`) |
| `compare_arms.py`, `compare_dirs.py` | Per-episode bitwise judges |
| `throughput_report.py`, `jobs_report.py`, `stats_report.py`, `expert_rep_report.py` | Throughput, cost tables and expert-record comparison |
| `full_estimate.py`, `build_measured.py`, `makespan_sim.py` | Full-benchmark estimate and its inputs; `makespan_sim.py` is the earlier schedule model |
| `render_bench.py`, `render_scale.py`, `render_static.py`, `render_concurrency.py`, `render_internals.py`, `move_cost.py`, `gpu_probe.py`, `probe_multienv.py`, `episode_overhead.py`, `substep_bench.py`, `infer_det.py`, `expert_det.py` | Profiling and reproducibility probes used in the rounds |
| `data/measured_final.json` | Per-task measured costs used by `full_estimate.py` |
| `data/robotwin_leaderboard_pi05.json` | Task list and official π0.5 co-train success rates from the public leaderboard |
| `autotune/pi05_h20_idle.textproto` | XLA autotune results recorded on an idle H20 (see below) |

Check that the two copies still match upstream except for the marked lines:

```bash
diff <(sed -n 772,975p scripts/eval_policy_xpolicylab.py) <(sed -n '/^def eval_remote_policy_opt/,$p' evalopt/opt_loop.py)
diff <(sed -n 1552,1740p envs/_base_task.py | sed 's/^    //') <(sed -n '/^def take_action_opt/,$p' evalopt/opt_take_action.py)
```

## Autotune file

`autotune/pi05_h20_idle.textproto` holds the XLA GEMM/convolution autotune results of the π0.5 server, recorded on an
idle H20 with JAX 0.5.3. `serve_policy.sh` loads it by default (`EVALOPT_AUTOTUNE`; an empty value disables it) so that
baseline and candidate servers make the same kernel choices. It is only valid for the same GPU model, driver and JAX
version; elsewhere, record a new file on an idle GPU by starting the server once with
`XLA_FLAGS=--xla_gpu_dump_autotune_results_to=<file>` and running one inference, then point `EVALOPT_AUTOTUNE` at it.
