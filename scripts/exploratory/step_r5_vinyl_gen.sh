#!/usr/bin/env bash
# R5 on vinyl: Gemma 31B final generation (batch 24).
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/overnight_2026-11-05/R5; mkdir -p $O
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || [ -d "$HOME/.cache/huggingface/hub/models--google--gemma-4-31B-it" ] || { echo "only ${avail} GB free"; exit 4; }
[[ -f $O/gemma4-31b/instruct_replies.csv.gz ]] || /opt/arena-env/bin/python -m extras.voice_fresh gen --family gemma4-31b
