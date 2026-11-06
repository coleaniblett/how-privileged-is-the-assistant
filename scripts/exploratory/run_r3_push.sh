#!/usr/bin/env bash
# R3 on marble: wait for vinyl's in-character judging (R3_IC_DONE) and Gemma wording 2 (GEMMA_W2_DONE; at 05:00 UTC push what exists), fetch, analyze, push once.
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/r3_push.lock; flock -n 7 || exit 0
O=results/overnight_2026-11-05/R3; [ -f $O/PUSHED ] && exit 0
DEADLINE=$(date -u -d "2026-11-06 05:00" +%s)
rv() { ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/$1" 2>/dev/null; }
until rv R3_IC_DONE && { rv GEMMA_W2_DONE || [ "$(date -u +%s)" -ge "$DEADLINE" ]; }; do sleep 60; done
nice -n 15 rsync -a --update -e ssh arena9-vinyl:/root/hpa-atlas2/$O/ $O/ --exclude='*.parquet' || true
nice -n 15 $PY -m extras.prefill_r3 analyze > /tmp/claude-0/r3_analyze.log 2>&1 || { tail -5 /tmp/claude-0/r3_analyze.log; exit 2; }
echo "$(date -u +%H:%M) R3 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R3 is the prefill belief result real (wording 2, Llama and Apertus, in-character check)" $O extras/prefill_r3.py scripts/exploratory && touch $O/PUSHED
