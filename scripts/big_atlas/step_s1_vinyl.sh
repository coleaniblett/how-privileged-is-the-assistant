#!/usr/bin/env bash
# spec S Stage 1 on vinyl: map (resumable chunks) -> analyze -> wait for the chosen splits -> reply check. GPU lock shared with the other chains.
cd /root/hpa-atlas2 || exit 1
export BA_SRC=/root/big_atlas_src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; M=results/big_atlas/private
mkdir -p $M results/big_atlas/stage1
[ -e $M/MAP_DONE ] || flock -x results/trait_atlas/.gpu.lock nice -n 15 $PY -m extras.big_atlas_map run || { echo FAIL > $M/MAP_FAILED; exit 1; }
nice -n 15 $PY -m extras.big_atlas_map analyze || { echo FAIL > $M/ANALYZE_FAILED; exit 1; }
touch $M/ANALYZE_DONE
for i in $(seq 1 1440); do [ -e data/big_atlas_splits.json ] && break; sleep 30; done
[ -e data/big_atlas_splits.json ] || exit 1
sleep 20
[ -e results/big_atlas/stage1/replycheck.json ] || flock -x results/trait_atlas/.gpu.lock nice -n 15 $PY -m extras.big_atlas_map replycheck || { echo FAIL > $M/REPLY_FAILED; exit 1; }
touch $M/REPLY_DONE
