#!/usr/bin/env bash
# R4 on marble: wait for vinyl's R4_JUDGED_DONE (and R2's judged files for the refusal judge rows), fetch, analyze, push once.
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/r4_push.lock; flock -n 7 || exit 0
O=results/overgeneralization/overnight_2026-11-05/R4; mkdir -p $O; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/R4_JUDGED_DONE" 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh --include='*/' --include='*_judged.csv.gz' --exclude='*' arena9-vinyl:/root/hpa-atlas2/$O/ $O/ || exit 1
nice -n 15 rsync -a -e ssh --include='judged_*.csv.gz' --exclude='*' arena9-vinyl:/root/r2_work/ /root/r2_work/ || true
nice -n 15 $PY -m extras.olmo_stages analyze > /tmp/claude-0/r4_analyze.log 2>&1 || { tail -5 /tmp/claude-0/r4_analyze.log; exit 2; }
echo "$(date -u +%H:%M) R4 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R4 which training stage does it (OLMo base, SFT, DPO, final; descriptive table)" $O/R4.md $O/R4_table.csv extras/olmo_stages.py scripts/exploratory && touch $O/PUSHED
