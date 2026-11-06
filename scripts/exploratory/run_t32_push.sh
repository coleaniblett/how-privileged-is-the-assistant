#!/usr/bin/env bash
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/t32_push.lock; flock -n 7 || exit 0
O=results/overnight_2026-11-05/R6; [ -f $O/T32_PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/R6B_JUDGED_DONE" 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh --include='*/' --include='base_judged.csv.gz' --exclude='*' arena9-vinyl:/root/hpa-atlas2/$O/ $O/ || exit 1
nice -n 15 $PY -m extras.role_says_breadth analyze > /tmp/claude-0/t32_analyze.log 2>&1 || { tail -n 5 /tmp/claude-0/t32_analyze.log; exit 2; }
git checkout results/role_says/role_says_replies_sample.csv.gz 2>/dev/null
printf '\n## 2026-11-06 (UTC) - Result: R6 base added (T3.2): Llama 3.1 8B and Apertus 8B base replies for spec Q; the BREADTH section of results/role_says/role_says.md now includes the in-character base column and the training-check rows. Final readings unchanged by construction.\n' >> docs/decisions.md
echo "$(date -u +%H:%M) T3.2 pushing" >> /root/overnight_log.md
push_paths "results: R6 base for Llama and Apertus added (T3.2)" results/role_says/role_says.md $O/role_says.md docs/decisions.md extras/role_says_breadth.py scripts/exploratory && touch $O/T32_PUSHED
