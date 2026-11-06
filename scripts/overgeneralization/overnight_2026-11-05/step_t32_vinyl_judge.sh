#!/usr/bin/env bash
# T3.2 judge on vinyl: base replies of Llama and Apertus (waits until 08:00 UTC), touches R6B_JUDGED_DONE.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/overgeneralization/overnight_2026-11-05/R6; DEADLINE=$(date -u -d "2026-11-06 08:00" +%s)
ready() { for f in $O/llama3.1-8b/base_replies.csv.gz $O/apertus-8b/base_replies.csv.gz; do [ -f $f ] || return 1; done; }
until ready || [ "$(date -u +%s)" -ge "$DEADLINE" ]; do sleep 60; done
exec 9>results/overgeneralization/trait_atlas/.gpu.lock; flock -x 9
RS_OUT=$O /opt/arena-env/bin/python -m extras.role_says_breadth judge; rc=$?; ready && [ $rc -eq 0 ] && touch $O/R6B_JUDGED_DONE; exit $rc
