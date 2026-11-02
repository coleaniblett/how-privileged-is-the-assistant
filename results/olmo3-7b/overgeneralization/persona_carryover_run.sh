#!/usr/bin/env bash
# Persona carryover on OLMo 3 7B, one stage at a time (disk holds one checkpoint): score, then delete the checkpoint.
# This machine's HF cache keeps file contents in a shared hub/blobs/ store (each blob has a .refs file naming the repos
# that use it), so deleting the repo folder alone frees nothing: blobs referenced only by that repo are removed too.
# Stages whose score CSV already exists are skipped.
set -euo pipefail
cd "$(dirname "$0")/../../.."
HUB="$HOME/.cache/huggingface/hub"
OUT=results/olmo3-7b/overgeneralization/persona_carryover
free_model() {
  local repo="models--${1//\//--}"
  rm -rf "$HUB/$repo"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi
  done
}
for stage in base sft dpo instruct; do
  model=$(uv run python -c "from core import config; print(config.MODELS['$stage'])" 2>/dev/null)
  if [[ -f "$OUT/$stage.csv" ]]; then echo "=== $stage: already scored, skipped"; continue; fi
  echo "=== $stage ($model) $(date)"
  uv run python -m measures.overgeneralization.persona_carryover score --model "$stage"
  free_model "$model"
  df -h / | tail -1
done
uv run python -m measures.overgeneralization.persona_carryover analyze
echo "=== done $(date)"
