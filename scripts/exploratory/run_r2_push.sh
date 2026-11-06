#!/usr/bin/env bash
# R2 on marble: wait for vinyl's R2_JUDGED_DONE, fetch judged files, analyze, push once (flock + marker).
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/r2_push.lock; flock -n 7 || exit 0
O=results/overnight_2026-11-05/R2; mkdir -p $O; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl 'test -f /root/r2_work/R2_JUDGED_DONE' 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh --include='judged_*.csv.gz' --exclude='*' arena9-vinyl:/root/r2_work/ /root/r2_work/ || exit 1
R2_LOCAL=/root/r2_work nice -n 15 $PY -m extras.refusal_judge analyze > /tmp/claude-0/r2_analyze.log 2>&1 || { tail -5 /tmp/claude-0/r2_analyze.log; exit 2; }
echo "$(date -u +%H:%M) R2 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R2 did the reply help (judge vs regex refusal rates; validation, readings, slide D table)" $O/R2.md $O/R2_readings.csv $O/R2_slideD_table.csv extras/refusal_judge.py scripts/exploratory/run_r2_push.sh scripts/exploratory/step_r2_vinyl.sh scripts/exploratory/queue_driver.sh && touch $O/PUSHED
