#!/usr/bin/env bash
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
exec 7>/tmp/claude-0/t1_push.lock; flock -n 7 || exit 0
O=results/overgeneralization/overnight_2026-11-05/T1; mkdir -p $O /root/label_swap; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/T1_JUDGED_DONE" 2>/dev/null; do sleep 60; done
rsync -a -e ssh arena9-vinyl:/root/label_swap/replies/ /root/label_swap/replies/ || exit 1
rsync -a -e ssh arena9-vinyl:/root/hpa-atlas2/$O/ $O/ --include='*.csv' --include='*.txt' --include='*.sha256' --exclude='*' || true
nice -n 15 env LS_WORK=/root/label_swap $PY -m extras.label_swap analyze > /tmp/claude-0/t1_analyze.log 2>&1 || { tail -n 5 /tmp/claude-0/t1_analyze.log; exit 2; }
[ -f $O/N_USED.txt ] && { printf '\nTraining examples used: %s (frozen 20,000; projection in seconds for both arms: %s). Source of the frozen set: sha256 in `train.sha256`.\n' "$(grep LS_N $O/N_USED.txt | cut -d= -f2)" "$(grep proj $O/N_USED.txt | cut -d= -f2)" >> $O/T1.md; }
echo "$(date -u +%H:%M) T1 report written; pushing" >> /root/overnight_log.md
push_paths "results: spec R T1 same training different label (OLMo base fine-tuned with label assistant vs Morgan)" $O/T1.md $O/train_curve_A.csv $O/train_curve_B.csv $O/N_USED.txt $O/train.sha256 extras/label_swap.py scripts/overgeneralization && touch $O/PUSHED
