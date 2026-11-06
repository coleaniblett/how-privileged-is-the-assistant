#!/usr/bin/env bash
# T3.2 on marble: spec Q on Llama and Apertus BASE (the R6 training check), batch 32; copies replies to vinyl.
cd /root/hpa-main-wt; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/R6
gpu_wait main-session "T3.2 marble"; trap gpu_release EXIT
for fam in llama3.1-8b apertus-8b; do
  [[ -f $O/$fam/base_replies.csv.gz ]] || RS_BATCH=32 retry $PY -m extras.role_says_breadth gen --family $fam --stage base || exit 1
  free_if_tight "$(stage_model $fam base)" 45
  ssh arena9-vinyl "mkdir -p /root/hpa-atlas2/$O/$fam"; rsync -a -e ssh $O/$fam/base_replies.csv.gz arena9-vinyl:/root/hpa-atlas2/$O/$fam/
done
touch $O/MARBLE_BASE_DONE
