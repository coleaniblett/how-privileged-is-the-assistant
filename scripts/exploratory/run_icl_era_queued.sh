#!/usr/bin/env bash
# Spec E (in-context era shift) on OLMo 3 7B, one stage at a time: wait for the GPU lock, check >= 15 GB free before each checkpoint download,
# score the stage, free its checkpoint, then analyze. Probe list and plan: docs/decisions.md (frozen 2026-11-03 before scoring).
#   bash scripts/exploratory/run_icl_era_queued.sh        # STAGES="base sft dpo instruct" by default; DRY_RUN=1 prints the plan
set -u
cd "$(dirname "$0")/../.."
PY=/opt/arena-env/bin/python
STAGES="${STAGES:-base instruct sft dpo}"
DRY="${DRY_RUN:-0}"
HUB="$HOME/.cache/huggingface/hub"
free_model() {
  local repo="models--${1//\//--}"
  rm -rf "$HUB/$repo"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi
  done
}
if [[ "$DRY" != "1" ]]; then
  while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
  echo "owner=main-session purpose=icl_era (spec E) OLMo ($STAGES); released when run_icl_era_queued.sh ends" > /tmp/claude-0/gpu.lock
  trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
fi
echo "start: $(date -u +%T); stages: $STAGES"
for stage in $STAGES; do
  model=$($PY -c "from core import config; print(config.FAMILIES['olmo3-7b']['stages']['$stage'])")
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  echo "=== $stage ($model): ${free_gb} GB free $(date -u +%T)"
  if [[ "$DRY" == "1" ]]; then echo "[dry] $PY -m extras.icl_era score --family olmo3-7b --model $stage"; continue; fi
  if [ "$free_gb" -lt 15 ]; then echo "less than 15 GB free: stopping"; break; fi
  $PY -m extras.icl_era score --family olmo3-7b --model "$stage" || { echo "score failed for $stage"; break; }
  free_model "$model"
done
[[ "$DRY" == "1" ]] || $PY -m extras.icl_era analyze
echo "done: $(date -u +%T)"
