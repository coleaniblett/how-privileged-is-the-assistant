#!/usr/bin/env bash
# T3.3 on vinyl: T1 second seed (LS_SEED=1: different example order and initialisation seed; base and official SFT replies reused from seed 0). Checkpoints are deleted after generation to keep disk for the judge.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LS_WORK=/root/label_swap_s1 LS_SEED=1 LS_OUT=results/overnight_2026-11-05/T1/seed1
PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
O=$LS_OUT; mkdir -p $O $LS_WORK/replies
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
cp -n /root/label_swap/train.jsonl /root/label_swap/train.sha256 $LS_WORK/; for f in base sft; do cp -n /root/label_swap/replies/$f.csv.gz /root/label_swap/replies/${f}_judged.csv.gz $LS_WORK/replies/ 2>/dev/null; done
mdl() { $PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['$1'])"; }
free_model "$(mdl instruct)"; rm -rf /root/label_swap/ckpt_A
for arm in A B; do
  avail=$(df --output=avail -BG / | tail -n 1 | tr -dc 0-9); [ "$avail" -ge 45 ] || { echo "only ${avail} GB free"; exit 4; }
  [ -f $LS_WORK/replies/$arm.csv.gz ] || { $PY -m extras.label_swap train --arm $arm && $PY -m extras.label_swap gen --tag $arm || exit 1; }
  rm -rf $LS_WORK/ckpt_$arm
done
free_model allenai/Olmo-3-1025-7B
$PY -m extras.label_swap judge && touch $O/T33_JUDGED_DONE
