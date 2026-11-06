#!/usr/bin/env bash
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/t33_push.lock; flock -n 7 || exit 0
O=results/overnight_2026-11-05/T1/seed1; mkdir -p $O /root/label_swap_s1/replies; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/T33_JUDGED_DONE" 2>/dev/null; do sleep 60; done
rsync -a -e ssh arena9-vinyl:/root/label_swap_s1/replies/ /root/label_swap_s1/replies/ || exit 1
rsync -a -e ssh arena9-vinyl:/root/hpa-atlas2/$O/ $O/ --include='*.csv' --exclude='*' || true
nice -n 15 env LS_WORK=/root/label_swap_s1 LS_SEED=1 LS_OUT=$O $PY -m extras.label_swap analyze > /tmp/claude-0/t33_analyze.log 2>&1 || { tail -n 5 /tmp/claude-0/t33_analyze.log; exit 2; }
printf '\nSecond seed (T3.3): seed 1 for example order and initialisation; base and official SFT replies reused from the first run; same frozen 20,000 examples. Compare with `../T1.md`.\n' >> $O/T1.md
echo "$(date -u +%H:%M) T3.3 second seed pushing" >> /root/overnight_log.md
push_paths "results: T1 second seed (T3.3)" $O/T1.md $O/train_curve_A.csv $O/train_curve_B.csv extras/label_swap.py scripts/exploratory && touch $O/PUSHED
