#!/usr/bin/env bash
# Spec G trait atlas (DO NOT RUN before Adeeb's go; docs/decisions.md 2026-11-04). One checkpoint at a time, generation then detectors, then the local judge
# (Gemma 4 31B instruct, kept on disk for it), hand-check sheet and analysis. Intended for arena9-vinyl (96 GB GPU, ~90 GB disk): all weights on the GPU.
# Every checkpoint is freed with its blobs (free_model); google/gemma-4-31B-it is never freed here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() {   # folder AND blobs: rm -rf of the folder alone leaves the checkpoint's blobs behind
  local repo="models--${1//\//--}"
  rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi
  done
}
KEEP="google/gemma-4-31B-it"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do echo "GPU busy, waiting"; sleep 60; done
for spec in "olmo3-7b base" "olmo3-7b sft" "olmo3-7b dpo" "olmo3-7b instruct" "qwen3-8b base" "qwen3-8b instruct" "gemma4-31b base" "gemma4-31b instruct"; do
  set -- $spec; fam=$1; stage=$2
  [[ -f "results/trait_atlas/$fam/${stage}_replies.jsonl.gz" ]] && { echo "=== $fam $stage: done, skipped"; continue; }
  model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "less than 70 GB free for $model"; exit 3; }
  echo "=== $fam $stage ($model) $(date -u +%T)"
  $PY -m extras.trait_atlas generate --family $fam --model $stage
  [ "$model" = "$KEEP" ] || free_model "$model"
done
$PY -m extras.trait_atlas detect
echo "=== judge $(date -u +%T)"
$PY -m extras.trait_atlas judge
free_model "$KEEP"
$PY -m extras.trait_atlas handcheck
$PY -m extras.trait_atlas analyze
echo "=== done $(date -u +%T)"
