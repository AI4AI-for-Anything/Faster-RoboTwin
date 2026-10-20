#!/usr/bin/env bash
# Summarise a scale_test.sh output directory: per K, wall, per-process get_obs median, pixel hashes, GPU util.
# usage: scale_report.sh <scale_test_out_dir>
R=$1
for d in $R/k*; do
  [ -f $d/t_end ] || continue
  echo "== $(basename $d) wall=$(python3 -c "print(round($(cat $d/t_end)-$(cat $d/t_start),1))")s"
  cat $d/p*.json | python3 -c "
import sys,json
rs=[json.loads(l) for l in sys.stdin.read().replace('}{','}\n{').splitlines()]
print(' n',len(rs),'get_obs_med_ms',sorted(r['get_obs_ms_median'] for r in rs),'act_ms',rs[0]['take_action_ms_median'],'sha',sorted(set(r['image_seq_sha'] for r in rs)),'cpu/wall',[round(r['cpu_s']/r['wall_s'],2) for r in rs][:4])"
  awk -F, '{gsub(/ %/,"",$2); s+=$2; n++} END {print " gpu util mean", s/n}' $d/gpu.csv
done
