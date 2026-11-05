#!/usr/bin/env bash
# R5 on marble: wait for vinyl's R5_JUDGED_DONE (Gemma replies and the in-character sample for all three families), fetch Gemma replies and judged files, analyze, push once.
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/r5_push.lock; flock -n 7 || exit 0
O=results/overnight_2026-11-05/R5; mkdir -p $O; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/R5_JUDGED_DONE" 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh --include='*/' --include='instruct_replies.csv.gz' --include='instruct_judged.csv.gz' --exclude='*' arena9-vinyl:/root/hpa-atlas2/$O/ $O/ || exit 1
nice -n 15 $PY -m extras.voice_fresh analyze > /tmp/claude-0/r5_analyze.log 2>&1 || { tail -5 /tmp/claude-0/r5_analyze.log; exit 2; }
echo "$(date -u +%H:%M) R5 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R5 voice on fresh questions (strict layout, stock phrases, in-character check; OLMo, Qwen, Gemma 31B)" $O/R5.md $O/R5_readings.csv $O/R5_replies_sample.csv.gz extras/voice_fresh.py scripts/exploratory && touch $O/PUSHED
