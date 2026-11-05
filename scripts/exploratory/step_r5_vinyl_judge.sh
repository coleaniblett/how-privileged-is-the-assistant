#!/usr/bin/env bash
# R5 in-character judge (30% sample) for every replies file on vinyl; touches R5_JUDGED_DONE only when all three families' replies were present.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/overnight_2026-11-05/R5
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
for f in olmo3-7b qwen3-8b gemma4-31b; do [ -f $O/$f/instruct_replies.csv.gz ] || { echo "missing $f replies"; exit 3; }; done
/opt/arena-env/bin/python -m extras.voice_fresh judge && touch $O/R5_JUDGED_DONE
