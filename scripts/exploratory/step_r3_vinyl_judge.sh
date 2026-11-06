#!/usr/bin/env bash
# R3c on vinyl: in-character judge on all saved final continuations (new-persona conditions), waiting for marble's continuations (until 04:00 UTC).
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/overgeneralization/overnight_2026-11-05/R3
DEADLINE=$(date -u -d "2026-11-06 04:00" +%s)
until [ -f $O/CONT_SYNCED ] || [ "$(date -u +%s)" -ge "$DEADLINE" ]; do sleep 60; done
# wording 1 continuations for OLMo and Qwen live in spec O's folder on marble; vinyl has Gemma's own. Fetch OLMo/Qwen w1 copies if they were synced.
exec 9>results/overgeneralization/trait_atlas/.gpu.lock; flock -x 9
/opt/arena-env/bin/python -m extras.prefill_r3 judge && touch $O/R3_IC_DONE
