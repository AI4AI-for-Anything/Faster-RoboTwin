# Performance figures

Regenerate all figures with `python docs/performance-views/render.py` (matplotlib only, no GPU or simulator).

- `01-estimate.svg`: full-benchmark **estimate** from `evalopt/full_estimate.py` ([estimate.json](estimate.json)):
  official pipeline 73.5 h vs. Faster-RoboTwin 16.8 h of wall clock on 8× H20 (4.38×), GPU time 538.8 → 127.9 GPU-h
  (4.21×). Not a measured full run: 6 of 100 jobs measured, 8% of the steps.
- `02-ladder.svg`: top, **measured** steady-state throughput per GPU on place_container_plate / clean for the
  cumulative configurations ([ladder.json](ladder.json), rounds R06, R11, R12; official baseline from the 10-episode
  official arm of R12, including per-episode expert-check and setup costs). The hatched bar is the GPU-bound ceiling
  derived from the R19 frame log (98.4 ms of GPU work per step). Bottom, per-task **estimates**: GPU time of one
  100-episode job, official ÷ Faster-RoboTwin, from the same job model as `full_estimate.py`.
- `03-time-budget.svg`: **measured** cost breakdowns ([time-budget.json](time-budget.json)): per step for one job alone on
  a GPU (stack_blocks_three / clean, R05) and fixed start-up per job (R05, R07).
- `04-ceiling.svg`: per-step GPU work at 14 jobs per GPU on place_container_plate / clean (R19 frame log, measured)
  against the measured step interval and a 5× pace, and the full-benchmark **estimate** with the 5× line and the
  ≈4.6× ceiling from R18. The deferred-video note refers to the optional mode, not the default.
- `../change-views/module-map.svg`: the switches grouped by what they change, with their files.

Every number in these figures is also in the README tables or in [measurement](../measurement.md).
