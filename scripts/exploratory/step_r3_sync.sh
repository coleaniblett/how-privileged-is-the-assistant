#!/usr/bin/env bash
# marble: after R3's chain finished, copy its continuations to vinyl for the in-character judge.
cd /root/hpa-main-wt; O=results/overnight_2026-11-05/R3
until [ -f $O/MARBLE_DONE ]; do sleep 60; done
ssh arena9-vinyl "mkdir -p /root/hpa-atlas2/$O"
rsync -a -e ssh $O/ arena9-vinyl:/root/hpa-atlas2/$O/ --include='*/' --include='*_continuations.csv' --exclude='*'
for f in olmo3-7b qwen3-8b; do ssh arena9-vinyl "mkdir -p /root/hpa-atlas2/results/belief_induction/arm2/$f"; rsync -a -e ssh results/belief_induction/arm2/$f/instruct_continuations.csv arena9-vinyl:/root/hpa-atlas2/results/belief_induction/arm2/$f/; done
touch $O/CONT_SYNCED
