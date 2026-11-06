#!/usr/bin/env bash
# T1 on vinyl: timing sample, then arm A and B training (full fine-tune), generation for base, official SFT, A, B, then the judge for denies_feelings. Keeps final checkpoints on vinyl (not in git).
# Projection over 4 h for both arms -> LS_N=10000 examples (said in the report). Log /root/atlas_logs/t1.log; markers T1_GEN_DONE and T1_JUDGED_DONE in results/overgeneralization/overnight_2026-11-05/T1.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LS_WORK=/root/label_swap
PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/T1; mkdir -p $O /root/label_swap
exec 9>results/overgeneralization/trait_atlas/.gpu.lock; flock -x 9
mdl() { $PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['$1'])"; }
[ -f /root/label_swap/train.jsonl ] || { echo "missing training data"; exit 3; }
free_model "$(mdl instruct)"
avail=$(df --output=avail -BG / | tail -n 1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "only ${avail} GB free"; exit 4; }
if [ ! -f $O/train_curve_A_timing.csv ]; then $PY -m extras.label_swap train --arm A --max-steps 20 || exit 1; fi
proj=$($PY -c "import pandas as pd;c=pd.read_csv('$O/train_curve_A_timing.csv');print(int(c.seconds.iloc[-1]/len(c)*1250*2))")
echo "projected both arms: ${proj} s"
if [ "$proj" -gt 14400 ]; then export LS_N=10000; echo "LS_N=10000 (projection over 4 h)"; fi
echo "LS_N=${LS_N:-20000}" > $O/N_USED.txt; echo "projection_s=$proj" >> $O/N_USED.txt
for arm in A B; do [ -f /root/label_swap/ckpt_$arm/config.json ] || $PY -m extras.label_swap train --arm $arm || exit 1; done
cp /root/label_swap/train.sha256 $O/ 2>/dev/null
for tag in A B base sft; do [ -f /root/label_swap/replies/$tag.csv.gz ] || $PY -m extras.label_swap gen --tag $tag || exit 1; done
touch $O/T1_GEN_DONE
free_model allenai/Olmo-3-1025-7B; free_model allenai/Olmo-3-7B-Instruct-SFT
avail=$(df --output=avail -BG / | tail -n 1 | tr -dc 0-9)
if [ "$avail" -lt 70 ]; then echo "only ${avail} GB free for the judge: removing checkpoint B (replies kept)"; rm -rf /root/label_swap/ckpt_B; fi
$PY -m extras.label_swap judge && touch $O/T1_JUDGED_DONE
