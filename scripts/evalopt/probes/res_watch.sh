#!/usr/bin/env bash
# Sample per-process resources of every simulator worker (prof_eval.py) whose command line contains
# <pattern>: open fds, memory maps, threads, RSS. One line per process per sample.
# usage: res_watch.sh <pattern> <out.tsv> <seconds> [interval]
PAT=$1; OUT=$2; DUR=$3; IV=${4:-15}
end=$(( $(date +%s) + DUR ))
echo -e "t\tpid\tfds\tmaps\tthreads\trss_mb" > "$OUT"
while [ "$(date +%s)" -lt "$end" ]; do
  for p in $(pgrep -f "prof_eval.py.*$PAT"); do
    [ -d /proc/$p ] || continue
    f=$(ls /proc/$p/fd 2>/dev/null | wc -l); m=$(wc -l < /proc/$p/maps 2>/dev/null)
    th=$(awk '/^Threads/{print $2}' /proc/$p/status 2>/dev/null); r=$(awk '/^VmRSS/{print int($2/1024)}' /proc/$p/status 2>/dev/null)
    echo -e "$(date +%s)\t$p\t$f\t$m\t$th\t$r" >> "$OUT"
  done
  sleep "$IV"
done
