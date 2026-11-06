#!/usr/bin/env bash
# Confirmatory persona-carryover run on Qwen3-8B base and instruct (README criterion 7, breadth arm): the SAME frozen item set and
# analysis as the OLMo run (scripts/exploratory/run_carryover_confirmatory.sh), --family qwen3-8b, assistant-slot empty think block ON
# (the pre-declared primary rendering). Writes only under results/overgeneralization/persona_carryover/qwen3-8b/confirmatory/. FROZEN_SHA256 must match.
#   FROZEN_SHA256=<sha256 of the item file> bash scripts/exploratory/run_carryover_confirmatory_qwen.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=data/persona_carryover_confirmatory.jsonl
OUT=results/overgeneralization/persona_carryover/qwen3-8b/confirmatory
PY=/opt/arena-env/bin/python
FAM=qwen3-8b
STAGES="${STAGES:-base instruct}"
actual=$(sha256sum "$DATA" | cut -d' ' -f1)
[[ "${FROZEN_SHA256:-}" == "$actual" ]] || { echo "NOT FROZEN: FROZEN_SHA256 must equal $actual"; exit 2; }
git diff --quiet HEAD -- "$DATA" && git ls-files --error-unmatch "$DATA" >/dev/null || { echo "$DATA is not committed/unmodified"; exit 2; }
HUB="$HOME/.cache/huggingface/hub"
free_model() {
  local repo="models--${1//\//--}"
  rm -rf "$HUB/$repo"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi
  done
}
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=confirmatory carryover Qwen3-8B ($STAGES); released when run_carryover_confirmatory_qwen.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
echo "dataset sha256 $actual; family $FAM; stages: $STAGES; outputs: $OUT"
mkdir -p "$OUT/persona_carryover"
for stage in $STAGES; do
  model=$($PY -c "from core import config; print(config.FAMILIES['$FAM']['stages']['$stage'])")
  if [[ -f "$OUT/persona_carryover/$stage.csv" ]]; then echo "=== $stage: already scored, skipped"; continue; fi
  echo "=== $stage ($model) $(date -u +%T)"
  $PY -m measures.overgeneralization.persona_carryover score --family $FAM --model "$stage" --stage "$stage" --dataset "$DATA" --out "$OUT/persona_carryover"
  free_model "$model"
done
$PY -m measures.overgeneralization.persona_carryover analyze --family $FAM --verdict --dataset "$DATA" --scores "$OUT/persona_carryover" --out-dir "$OUT"
echo "=== done $(date -u +%T)"
