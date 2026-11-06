#!/usr/bin/env bash
# R4 on marble: OLMo SFT and DPO new generation (L2 all, L3 capped at 1,500, Q subset on three sets), then copy the Q replies and the refusal-judge inputs to vinyl.
cd /root/hpa-main-wt; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/R4
gpu_wait main-session "spec R4 marble"; trap gpu_release EXIT
for st in sft dpo; do
  WML_BS=48 RS_BATCH=32 retry $PY -m extras.olmo_stages gen --stage $st || exit 1
  free_if_tight "$(stage_model olmo3-7b $st)" 50
done
R2_LOCAL=/root/r2_work $PY -m extras.refusal_judge prepare_stages
ssh arena9-vinyl 'mkdir -p /root/hpa-atlas2/results/overgeneralization/overnight_2026-11-05/R4 /root/r2_work'
rsync -a -e ssh $O/ arena9-vinyl:/root/hpa-atlas2/$O/ --include='*/' --include='*_replies.csv.gz' --exclude='*'
rsync -a -e ssh /root/r2_work/inputs_olmo_stages.csv.gz arena9-vinyl:/root/r2_work/
touch $O/MARBLE_REPLIES_DONE
