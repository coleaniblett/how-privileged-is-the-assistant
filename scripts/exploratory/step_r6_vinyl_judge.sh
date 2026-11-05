#!/usr/bin/env bash
# R6 judge on vinyl (spec Q's judge for denies_feelings, corrects_false_premise, hedging_unanswerable + 30% in-character sample), then touches R6_JUDGED_DONE.
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/overnight_2026-11-05/R6
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
for f in llama3.1-8b apertus-8b; do [ -f $O/$f/instruct_replies.csv.gz ] || { echo "missing $f replies"; exit 3; }; done
RS_OUT=$O /opt/arena-env/bin/python -m extras.role_says_breadth judge && touch $O/R6_JUDGED_DONE
