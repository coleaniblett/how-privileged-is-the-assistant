#!/usr/bin/env bash
# One checkpoint at a time: wait for a free GPU, check disk, score (downloads if needed), then remove the checkpoint
# from the HF cache only if this script downloaded it. Gemma 4 31B is not included (its IT checkpoint is not on disk;
# the base copy in /dev/shm belongs to a paused collection and is not touched).
#   bash exploration/turns_test/run_all.sh 2>&1 | tee exploration/turns_test/results/run.log
set -euo pipefail
PY=${PY:-/opt/arena-env/bin/python}
cd "$(dirname "$0")/../.."
for spec in olmo3-7b:base:allenai/Olmo-3-1025-7B olmo3-7b:instruct:allenai/Olmo-3-7B-Instruct \
            qwen3-8b:base:Qwen/Qwen3-8B-Base qwen3-8b:instruct:Qwen/Qwen3-8B; do
  IFS=: read -r fam stage repo <<< "$spec"
  if [[ -f exploration/turns_test/results/$fam/$stage.csv ]]; then echo "skip $fam/$stage (done)"; continue; fi
  until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "$(date -u +%T) GPU busy, waiting"; sleep 60; done
  if $PY -m core.hf_cache cached "$repo"; then was_cached=1; else was_cached=0; fi
  $PY -m core.hf_cache check "$repo"
  echo "$(date -u +%T) start $fam/$stage"
  $PY -m exploration.turns_test.score --family "$fam" --stage "$stage"
  if [[ $was_cached == 0 ]]; then $PY -m core.hf_cache free "$repo"; fi
  df -h / | tail -1
done
echo "$(date -u +%T) all done"
