#!/usr/bin/env bash
# spec S Stage 1 on marble: wait for the vinyl map, pull its numbers, choose roles and traits (seed 0), push the choice, send the splits back, wait for the reply check, push it.
cd /root/hpa-main-wt || exit 1
source scripts/exploratory/lib_r.sh
export BA_SRC=/root/big_atlas_src; PY=/opt/arena-env/bin/python; V=arena9-vinyl; VR=/root/hpa-atlas2
until ssh $V "test -e $VR/results/big_atlas/private/ANALYZE_DONE || test -e $VR/results/big_atlas/private/MAP_FAILED || test -e $VR/results/big_atlas/private/ANALYZE_FAILED" 2>/dev/null; do sleep 60; done
ssh $V "test -e $VR/results/big_atlas/private/ANALYZE_DONE" || { echo "map failed" > results/big_atlas/stage1/FAILED; exit 1; }
rsync -a -e ssh $V:$VR/results/big_atlas/stage1/ results/big_atlas/stage1/
$PY -m extras.big_atlas_choose choose > results/big_atlas/stage1/choose.log 2>&1 || { echo "choose failed" > results/big_atlas/stage1/FAILED; exit 1; }
push_paths "spec S Stage 1: map numbers and the blind choice of 48 roles and 96 traits with discovery/confirmation splits (made before any Stage 2 code)" data/big_atlas_splits.json results/big_atlas/stage1/map_roles.csv results/big_atlas/stage1/map_traits.csv results/big_atlas/stage1/map_summary.json results/big_atlas/stage1/chosen_roles.csv results/big_atlas/stage1/chosen_traits.csv results/big_atlas/stage1/choose.log
rsync -a -e ssh data/big_atlas_splits.json $V:$VR/data/
until ssh $V "test -e $VR/results/big_atlas/private/REPLY_DONE || test -e $VR/results/big_atlas/private/REPLY_FAILED" 2>/dev/null; do sleep 60; done
rsync -a -e ssh $V:$VR/results/big_atlas/stage1/replycheck.json results/big_atlas/stage1/
push_paths "spec S Stage 1: reply-average check of the prompt-state map" results/big_atlas/stage1/replycheck.json
touch results/big_atlas/stage1/S1_CHAIN_DONE
