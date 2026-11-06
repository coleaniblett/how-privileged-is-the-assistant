#!/usr/bin/env bash
# R2 judging on vinyl (shared GPU lock; Gemma 31B-it judge). Inputs in /root/r2_work (rsynced from marble), outputs judged_*.csv.gz there; touches R2_JUDGED_DONE.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R2_LOCAL=/root/r2_work
exec 9>results/overgeneralization/trait_atlas/.gpu.lock; flock -x 9
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || [ -d "$HOME/.cache/huggingface/hub/models--google--gemma-4-31B-it" ] || { echo "only ${avail} GB free"; exit 4; }
/opt/arena-env/bin/python -m extras.refusal_judge judge && touch /root/r2_work/R2_JUDGED_DONE
