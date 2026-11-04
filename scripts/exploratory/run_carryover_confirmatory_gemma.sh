#!/usr/bin/env bash
# Confirmatory persona-carryover run on Gemma 4 31B base and instruct (README criterion 7, breadth arm): the SAME frozen item set and analysis
# as the OLMo and Qwen runs, --family gemma4-31b. bf16, batch 1, weights placed with device_map=auto (GPU_GIB GiB on the GPU, the rest in CPU RAM;
# no quantization). Assistant slot opens with the empty thought channel (core.render, docs/decisions.md 2026-11-02). Writes only under
# results/gemma4-31b/overgeneralization/confirmatory/. The instruct checkpoint cache (google/gemma-4-31B-it) is NEVER deleted; the base
# checkpoint is removed after scoring only if this run downloaded it (REMOVE_BASE=1, default).
#   FROZEN_SHA256=<sha256 of the item file> bash scripts/exploratory/run_carryover_confirmatory_gemma.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=data/persona_carryover_confirmatory.jsonl
OUT=results/gemma4-31b/overgeneralization/confirmatory
PY=/opt/arena-env/bin/python
FAM=gemma4-31b
STAGES="${STAGES:-base instruct}"
GPU_GIB="${GPU_GIB:-20}"
HUB="$HOME/.cache/huggingface/hub"
actual=$(sha256sum "$DATA" | cut -d' ' -f1)
[[ "${FROZEN_SHA256:-}" == "$actual" ]] || { echo "NOT FROZEN: FROZEN_SHA256 must equal $actual"; exit 2; }
git diff --quiet HEAD -- "$DATA" && git ls-files --error-unmatch "$DATA" >/dev/null || { echo "$DATA is not committed/unmodified"; exit 2; }
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=confirmatory carryover Gemma 4 31B ($STAGES); released when run_carryover_confirmatory_gemma.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
echo "dataset sha256 $actual; family $FAM; stages: $STAGES; gpu_gib $GPU_GIB; outputs: $OUT"
mkdir -p "$OUT/persona_carryover"
for stage in $STAGES; do
  model=$($PY -c "from core import config; print(config.FAMILIES['$FAM']['stages']['$stage'])")
  repo="models--${model//\//--}"
  had=0; [ -d "$HUB/$repo/snapshots" ] && [ -n "$(find "$HUB/$repo/blobs" -size +1G 2>/dev/null | head -1)" ] && had=1
  if [[ -f "$OUT/persona_carryover/$stage.csv" ]]; then echo "=== $stage: already scored, skipped"; continue; fi
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || [ "$had" = 1 ] || { echo "less than 70 GB free for the $stage download"; exit 3; }
  echo "=== $stage ($model) $(date -u +%T)"
  $PY -m measures.overgeneralization.persona_carryover score --family $FAM --model "$stage" --stage "$stage" --dataset "$DATA" --out "$OUT/persona_carryover" --gpu-gib "$GPU_GIB"
  if [ "$stage" = base ] && [ "$had" = 0 ] && [ "${REMOVE_BASE:-1}" = 1 ]; then rm -rf "${HUB:?}/${repo:?}"; echo "removed the base checkpoint this run downloaded"; fi
done
$PY -m measures.overgeneralization.persona_carryover analyze --family $FAM --verdict --dataset "$DATA" --scores "$OUT/persona_carryover" --out-dir "$OUT"
echo "=== done $(date -u +%T)"
