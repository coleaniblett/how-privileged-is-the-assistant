#!/usr/bin/env bash
# R4 judging on vinyl: spec Q's judge on the SFT/DPO Q subset (denies_feelings + 30% in-character sample) and R2's refusal judge on the SFT/DPO L2/L3 replies; touches R4_JUDGED_DONE.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R2_LOCAL=/root/r2_work
O=results/overnight_2026-11-05/R4
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
for s in sft dpo; do [ -f $O/olmo3-7b/${s}_replies.csv.gz ] || { echo "missing $s replies"; exit 3; }; done
[ -f /root/r2_work/inputs_olmo_stages.csv.gz ] || { echo "missing refusal inputs"; exit 3; }
RS_OUT=$O /opt/arena-env/bin/python -m extras.role_says judge && /opt/arena-env/bin/python -m extras.refusal_judge judge && touch $O/R4_JUDGED_DONE
