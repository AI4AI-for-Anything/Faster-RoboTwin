# skills

Priors for an agent that optimizes a closed-loop policy evaluation. These four cards are the evaluation-side
counterparts of the `train-opt-*` training cards in
[ResearchBoost-Agent](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/main/skills); the
originals are not copied here, and these evaluation cards are not in ResearchBoost-Agent yet. Copy the relevant
cards into the agent's skill directory (for Codex: `$CODEX_HOME/skills/<name>/SKILL.md`) and name them in the
task statement. The numbers come from RoboTwin `ea8b211` + the official π0.5 co-train checkpoint on 8× H20;
re-verify them on your setup before treating them as facts. The verdict is always the per-episode comparison
(`evalopt/compare_arms.py`).

| Card | Read when |
|---|---|
| `eval-opt-equivalence` | before changing the evaluation flow or schedule, or skipping any computation: E0–E3 tiers, reference reproducibility (pinned autotune), per-episode fingerprints, random-number chains, stateful-policy cross-talk, render statelessness |
| `eval-opt-tiered-profiling` | before choosing a hotspot: job → episode → step → camera accounting, concurrency-scaling ablation, the three hotspot kinds |
| `eval-opt-measurement` | before timing or comparing anything: valid-run criteria, reference–candidate order, rendering time-sharing, full-benchmark estimates, port and script hygiene on shared hosts |
| `eval-opt-action-ladder` | deciding what to try: L0 scheduling and resident processes → L1 removing wasted work → L2 caching policy-independent results → L3 loop changes, with tiers, evidence and what is not done |
