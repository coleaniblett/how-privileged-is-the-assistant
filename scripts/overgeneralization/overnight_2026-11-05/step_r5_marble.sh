#!/usr/bin/env bash
# R5 on marble: OLMo and Qwen final generation (spec R5), then copy the replies to vinyl for the in-character judge.
cd /root/hpa-main-wt; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/R5
gpu_wait main-session "spec R5 marble"; trap gpu_release EXIT
for fam in olmo3-7b qwen3-8b; do
  [[ -f $O/$fam/instruct_replies.csv.gz ]] || RS_BATCH=32 retry $PY -m extras.voice_fresh gen --family $fam || exit 1
done
ssh arena9-vinyl 'mkdir -p /root/hpa-atlas2/results/overgeneralization/overnight_2026-11-05/R5'
rsync -a -e ssh $O/ arena9-vinyl:/root/hpa-atlas2/$O/ --include='*/' --include='instruct_replies.csv.gz' --exclude='*'
touch $O/MARBLE_REPLIES_DONE
