#!/usr/bin/env bash
# T2 judge on vinyl: waits for marble's replies (until 07:00 UTC), judges the judged traits, touches T2_JUDGED_DONE.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/overnight_2026-11-05/T2; DEADLINE=$(date -u -d "2026-11-06 07:00" +%s)
ready() { for f in $O/olmo3-7b/instruct_replies.csv.gz $O/qwen3-8b/instruct_replies.csv.gz; do [ -f $f ] || return 1; done; }
until ready || [ "$(date -u +%s)" -ge "$DEADLINE" ]; do sleep 60; done
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
RO_OUT=$O /opt/arena-env/bin/python -m extras.role_obey judge; rc=$?; ready && [ $rc -eq 0 ] && touch $O/T2_JUDGED_DONE; exit $rc
