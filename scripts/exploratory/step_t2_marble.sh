#!/usr/bin/env bash
# T2 on marble: OLMo and Qwen final generation (8,640 replies each, batch 32), then copy to vinyl for the judge.
cd /root/hpa-main-wt; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/T2
gpu_wait main-session "T2 marble"; trap gpu_release EXIT
for fam in olmo3-7b qwen3-8b; do
  [[ -f $O/$fam/instruct_replies.csv.gz ]] || RS_BATCH=32 retry $PY -m extras.role_obey gen --family $fam || exit 1
  ssh arena9-vinyl "mkdir -p /root/hpa-atlas2/$O/$fam"; rsync -a -e ssh $O/$fam/instruct_replies.csv.gz arena9-vinyl:/root/hpa-atlas2/$O/$fam/
done
touch $O/MARBLE_REPLIES_DONE
