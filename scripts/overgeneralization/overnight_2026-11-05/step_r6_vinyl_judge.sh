#!/usr/bin/env bash
# R6 judge on vinyl (spec Q's judge + 30% in-character sample). Waits (polling, outside the GPU lock) for marble's replies until 04:00 UTC (spec R: judge what exists by then), then judges; touches the DONE marker only when everything expected was present.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R2_LOCAL=/root/r2_work
O=results/overgeneralization/overnight_2026-11-05/R6
DEADLINE=$(date -u -d "2026-11-06 04:00" +%s)
ready() { for f in $O/llama3.1-8b/instruct_replies.csv.gz $O/apertus-8b/instruct_replies.csv.gz; do [ -f $f ] || return 1; done; }
until ready || [ "$(date -u +%s)" -ge "$DEADLINE" ]; do sleep 60; done
exec 9>results/overgeneralization/trait_atlas/.gpu.lock; flock -x 9
RS_OUT=$O /opt/arena-env/bin/python -m extras.role_says_breadth judge
rc=$?; ready && [ $rc -eq 0 ] && touch $O/R6_JUDGED_DONE; exit $rc
