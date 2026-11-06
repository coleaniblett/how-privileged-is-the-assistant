#!/usr/bin/env bash
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
exec 7>/tmp/claude-0/t2_push.lock; flock -n 7 || exit 0
O=results/overgeneralization/overnight_2026-11-05/T2; mkdir -p $O; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/T2_JUDGED_DONE" 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh --include='*/' --include='*_judged.csv.gz' --exclude='*' arena9-vinyl:/root/hpa-atlas2/$O/ $O/ || exit 1
nice -n 15 $PY -m extras.role_obey analyze > /tmp/claude-0/t2_analyze.log 2>&1 || { tail -n 5 /tmp/claude-0/t2_analyze.log; exit 2; }
echo "$(date -u +%H:%M) T2 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R T2 role says yes or no on each trait (descriptive map; OLMo, Qwen)" $O/T2.md $O/T2_map.csv extras/role_obey.py scripts/overgeneralization && touch $O/PUSHED
