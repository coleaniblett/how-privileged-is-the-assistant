#!/usr/bin/env bash
# Gemma 4 31B confirmatory carryover RERUN with the planned empty thought channel on the final assistant turn (--assistant-opening), base then instruct.
# Same frozen items (FROZEN_SHA256), same criteria and verdict code. Outputs ONLY under results/gemma4-31b/overgeneralization/confirmatory_thought_channel/
# (the first run, confirmatory/, is untouched). All weights on the GPU (GPU_GIB, default 90), bf16, batch 1. Disk: base and instruct do not fit together, so the instruct
# checkpoint is freed (folder and blobs) for base, then re-downloaded and KEPT at the end for the atlas judge.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1
DATA=data/persona_carryover_confirmatory.jsonl
OUT=results/gemma4-31b/overgeneralization/confirmatory_thought_channel
PY=/opt/arena-env/bin/python
FAM=gemma4-31b
GPU_GIB="${GPU_GIB:-90}"
HUB="$HOME/.cache/huggingface/hub"
actual=$(sha256sum "$DATA" | cut -d' ' -f1)
[[ "${FROZEN_SHA256:-}" == "$actual" ]] || { echo "NOT FROZEN: FROZEN_SHA256 must equal $actual"; exit 2; }
git diff --quiet HEAD -- "$DATA" && git ls-files --error-unmatch "$DATA" >/dev/null || { echo "$DATA is not committed/unmodified"; exit 2; }
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do echo "GPU busy, waiting"; sleep 30; done
echo "dataset sha256 $actual; family $FAM; assistant-opening ON; outputs: $OUT"
mkdir -p "$OUT/persona_carryover"
for stage in base instruct; do
  model=$($PY -c "from core import config; print(config.FAMILIES['$FAM']['stages']['$stage'])")
  [[ -f "$OUT/persona_carryover/$stage.csv" ]] && { echo "=== $stage: already scored, skipped"; continue; }
  if [ "$stage" = base ]; then free_model "google/gemma-4-31B-it"; fi      # room for base; instruct is re-downloaded below
  echo "=== $stage ($model) $(date -u +%T)"
  $PY -m measures.overgeneralization.persona_carryover score --family $FAM --model "$stage" --stage "$stage" --dataset "$DATA" --out "$OUT/persona_carryover" --gpu-gib "$GPU_GIB" --assistant-opening
  if [ "$stage" = base ]; then free_model "$model"; fi
done
$PY -m measures.overgeneralization.persona_carryover analyze --family $FAM --verdict --assistant-opening --dataset "$DATA" --scores "$OUT/persona_carryover" --out-dir "$OUT"
echo "=== done $(date -u +%T)"
