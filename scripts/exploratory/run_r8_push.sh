#!/usr/bin/env bash
# R8 on marble: wait for vinyl's R8_JUDGED_DONE, fetch judged files, analyze, push once (the blind subagent addendum is added afterwards).
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/r8_push.lock; flock -n 7 || exit 0
O=results/overgeneralization/overnight_2026-11-05/R8; mkdir -p $O; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/R8_JUDGED_DONE" 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh --include='judged_*.csv.gz' --include='inputs_r8_*.csv.gz' --exclude='*' arena9-vinyl:/root/r2_work/ /root/r2_work/ || exit 1
rsync -a -e ssh arena9-vinyl:/root/hpa-atlas2/$O/gen/ $O/gen/ || true
R2_LOCAL=/root/r2_work nice -n 15 $PY -m extras.refusal_256 prepare >/dev/null 2>&1
R2_LOCAL=/root/r2_work nice -n 15 $PY -m extras.refusal_256 analyze > /tmp/claude-0/r8_analyze.log 2>&1 || { tail -n 5 /tmp/claude-0/r8_analyze.log; exit 2; }
echo "$(date -u +%H:%M) R8 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R8 L2 at 256 tokens judged (did the reply help, with room to finish)" $O/R8.md $O/R8_cells.csv extras/refusal_256.py extras/refusal_judge.py scripts/exploratory && touch $O/PUSHED
