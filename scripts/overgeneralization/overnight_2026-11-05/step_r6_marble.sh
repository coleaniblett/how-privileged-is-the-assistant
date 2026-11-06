#!/usr/bin/env bash
# R6 on marble: spec Q on Llama 3.1 8B and Apertus 8B, final only, batch 32; then copy replies to vinyl for the judge.
cd /root/hpa-main-wt; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/R6
gpu_wait main-session "spec R6 marble"; trap gpu_release EXIT
for fam in llama3.1-8b apertus-8b; do
  [[ -f $O/$fam/instruct_replies.csv.gz ]] || RS_BATCH=32 retry $PY -m extras.role_says_breadth gen --family $fam || exit 1
  free_if_tight "$(stage_model $fam instruct)" 40
done
ssh arena9-vinyl 'mkdir -p /root/hpa-atlas2/results/overgeneralization/overnight_2026-11-05/R6'
rsync -a -e ssh $O/ arena9-vinyl:/root/hpa-atlas2/$O/ --include='*/' --include='instruct_replies.csv.gz' --exclude='*'
touch $O/MARBLE_REPLIES_DONE
