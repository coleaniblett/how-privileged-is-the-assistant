#!/usr/bin/env bash
# R5 in-character judge (30% sample) on vinyl. Waits (polling, outside the GPU lock) for marble's replies until 04:00 UTC (spec R: judge what exists by then), then judges; touches the DONE marker only when everything expected was present.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R2_LOCAL=/root/r2_work
O=results/overgeneralization/overnight_2026-11-05/R5
DEADLINE=$(date -u -d "2026-11-06 04:00" +%s)
ready() { for f in $O/olmo3-7b/instruct_replies.csv.gz $O/qwen3-8b/instruct_replies.csv.gz $O/gemma4-31b/instruct_replies.csv.gz; do [ -f $f ] || return 1; done; }
until ready || [ "$(date -u +%s)" -ge "$DEADLINE" ]; do sleep 60; done
exec 9>results/overgeneralization/trait_atlas/.gpu.lock; flock -x 9
/opt/arena-env/bin/python -m extras.voice_fresh judge
rc=$?; ready && [ $rc -eq 0 ] && touch $O/R5_JUDGED_DONE; exit $rc
