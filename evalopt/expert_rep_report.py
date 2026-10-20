"""Compare expert-check records (expert_cache.py RECORD pickles) across runs, seed by seed.
usage: python expert_rep_report.py a.pkl b.pkl ...
"""
import json
import pickle
import sys

runs = {p: pickle.load(open(p, "rb"))["seeds"] for p in sys.argv[1:]}
seeds = sorted(set().union(*[set(r) for r in runs.values()]))
for s in seeds:
    row = []
    for p, r in runs.items():
        x = r.get(s)
        if x is None:
            row.append("-")
            continue
        acc = x.get("setup") == "ok" and x.get("play") == "ok" and x.get("plan_success") and x.get("check_success")
        row.append(f"{'ACC' if acc else 'rej'}(setup={x.get('setup')},play={x.get('play')},plan={x.get('plan_success')},chk={x.get('check_success')},info={json.dumps((x.get('info') or {}).get('info'), sort_keys=True)[:60]})")
    same = len(set(row)) == 1
    print(s, "SAME" if same else "DIFF")
    for p, x in zip(runs, row):
        print("   ", p.split("/")[-1], x)
