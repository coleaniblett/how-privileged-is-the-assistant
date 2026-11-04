#!/usr/bin/env bash
# Spec F-main: OLMo 3 7B base + instruct, then Qwen3-8B base + instruct (one checkpoint on disk at a time), then analyze.
# Materials must be committed first (data/persona_suppression_materials.json). Output: results/persona_suppression/.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/opt/arena-env/bin/python
git diff --quiet HEAD -- data/persona_suppression_materials.json && git ls-files --error-unmatch data/persona_suppression_materials.json >/dev/null || { echo "materials not committed"; exit 2; }
HUB="$HOME/.cache/huggingface/hub"
free_model() {
  local repo="models--${1//\//--}"
  rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi
  done
}
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec F-main persona suppression; released when run_persona_suppression.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for fam in olmo3-7b qwen3-8b; do
  for stage in base instruct; do
    model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
    if [[ -f "results/persona_suppression/$fam/${stage}_generations.jsonl" ]]; then echo "=== $fam $stage: done, skipped"; continue; fi
    avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 15 ] || { echo "less than 15 GB free"; exit 3; }
    echo "=== $fam $stage ($model) $(date -u +%T)"
    $PY -m extras.persona_suppression score --family $fam --model $stage
    free_model "$model"
  done
done
$PY -m extras.persona_suppression analyze
echo "=== done $(date -u +%T)"
