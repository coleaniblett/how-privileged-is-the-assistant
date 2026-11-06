#!/usr/bin/env bash
# R7: wait for marble's R7 files and vinyl's Gemma, fetch Gemma, analyze, push once.
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
exec 7>/tmp/claude-0/r7_push.lock; flock -n 7 || exit 0
O=results/overgeneralization/overnight_2026-11-05/R7; mkdir -p $O; [ -f $O/PUSHED ] && exit 0
until [ -f $O/MARBLE_DONE ] && ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/R7_GEMMA_DONE" 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh arena9-vinyl:/root/hpa-atlas2/$O/gemma4-31b/ $O/gemma4-31b/ || exit 1
nice -n 15 $PY -m extras.knows_says analyze > /tmp/claude-0/r7_analyze.log 2>&1 || { tail -5 /tmp/claude-0/r7_analyze.log; exit 2; }
echo "$(date -u +%H:%M) R7 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R7 knows versus says (forced answers and false-premise log-odds against free replies)" $O/R7.md $O/R7_readings.csv $O/olmo3-7b $O/qwen3-8b $O/gemma4-31b extras/knows_says.py scripts/overgeneralization && touch $O/PUSHED
