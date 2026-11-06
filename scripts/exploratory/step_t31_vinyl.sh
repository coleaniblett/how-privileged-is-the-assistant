#!/usr/bin/env bash
# T3.1 on vinyl: Q Gemma base (the training check dropped earlier): free the judge weights, download Gemma base, generate spec Q's replies (batch 24), free base, re-download the judge and judge the new file; touches T31_JUDGED_DONE.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 9>results/overgeneralization/trait_atlas/.gpu.lock; flock -x 9
mdl() { $PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['$1'])"; }
if [ ! -f results/overgeneralization/role_says/gemma4-31b/base_replies.csv.gz ]; then
  free_model "$(mdl instruct)"
  avail=$(df --output=avail -BG / | tail -n 1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "only ${avail} GB free"; exit 4; }
  $PY -m extras.role_says gen --family gemma4-31b --stage base || exit 1
  free_model "$(mdl base)"
fi
$PY -m extras.role_says judge && touch results/overgeneralization/role_says/T31_JUDGED_DONE
