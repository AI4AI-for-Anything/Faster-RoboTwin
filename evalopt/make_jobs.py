"""Write a jobs.json for prof_eval.py --jobs.
usage: python make_jobs.py <out.json> <port> <device_id> <job_root> TASK:CONFIG:N[:record|replay=PATH] ...
"""
import json
import os
import sys

CK = os.environ["PI05_CKPT"]  # official checkpoint, step directory 29999
out, port, dev, root = sys.argv[1:5]
jobs = []
for i, spec in enumerate(sys.argv[5:], 1):
    task, cfg, n, *rest = spec.split(":")
    job = {"out": f"{root}/j{i}_{task}_{cfg.replace('demo_', '')}",
           "args": ["--bench_name", "RoboTwin", "--task_name", task, "--env_cfg_type", "arx_x5",
                    "--policy_name", "Pi_05", "--host", "127.0.0.1", "--port", port, "--protocol", "ws",
                    "--device_id", dev, "--seed", "0", "--test_num", n, "--task_config", cfg,
                    "--expert_check", "true", "--eval_batch", "false",
                    "--additional_info", f"ckpt_name={CK},action_type=joint"]}
    for r in rest:
        k, v = r.split("=", 1)
        job[k] = v
    jobs.append(job)
json.dump(jobs, open(out, "w"), indent=1)
print(out, len(jobs))
