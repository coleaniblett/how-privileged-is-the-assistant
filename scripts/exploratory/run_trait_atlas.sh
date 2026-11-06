#!/usr/bin/env bash
# Spec G trait atlas, version 2 materials (go-ahead given 2026-11-04; docs/decisions.md). For arena9-vinyl (96 GB GPU, ~100 GB disk, shared): all weights on the GPU,
# one checkpoint at a time, HF_HUB_DISABLE_XET=1, every checkpoint freed with its blobs (free_model). google/gemma-4-31B-it is kept on disk for the judge.
# Disk order: with the 59 GB instruct checkpoint present only ~30 GB are free, so OLMo and Qwen run first, then Gemma instruct, the judge, and only then Gemma base:
# that one needs the space, so the instruct checkpoint is removed for it and re-downloaded (about 3 minutes) for the judge's second pass; it is on disk at the end.
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
wait_gpu() {   # the Gemma carryover has idle GPU gaps between its stages while a checkpoint downloads
  while pgrep -f "[r]un_carryover_confirmatory_gemma" >/dev/null || pgrep -f "[m]easures.overgeneralization.persona_carryover" >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do echo "GPU busy, waiting $(date -u +%T)"; sleep 60; done
}
gen() {   # fam stage need_gb
  local fam=$1 stage=$2 need=$3 model avail
  [[ -f "results/trait_atlas/$fam/${stage}_replies.jsonl.gz" ]] && { echo "=== $fam $stage: done, skipped"; return; }
  model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$avail" -lt "$need" ] && [ -d "$HUB/models--${KEEP//\//--}" ] && [ "$model" != "$KEEP" ]; then
    echo "only ${avail} GB free for $model: removing $KEEP (folder and blobs); it is re-downloaded for the judge"; free_model "$KEEP"
    avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  fi
  [ "$avail" -ge "$need" ] || { echo "less than $need GB free for $model"; exit 3; }
  echo "=== $fam $stage ($model) $(date -u +%T)"
  $PY -m extras.trait_atlas generate --family $fam --model $stage
  [ "$model" = "$KEEP" ] || free_model "$model"
  echo "=== $fam $stage done $(date -u +%T)"
}
wait_gpu
gen olmo3-7b base 20;  gen olmo3-7b sft 20;  gen olmo3-7b dpo 20;  gen olmo3-7b instruct 20
gen qwen3-8b base 20;  gen qwen3-8b instruct 20
gen gemma4-31b instruct 5
$PY -m extras.trait_atlas detect
echo "=== judge (first pass) $(date -u +%T)"; $PY -m extras.trait_atlas judge
gen gemma4-31b base 64
$PY -m extras.trait_atlas detect
echo "=== judge (second pass, Gemma base) $(date -u +%T)"; $PY -m extras.trait_atlas judge
$PY -m extras.trait_atlas handcheck
echo "=== analyze $(date -u +%T)"; $PY -m extras.trait_atlas analyze
echo "=== done $(date -u +%T)"
