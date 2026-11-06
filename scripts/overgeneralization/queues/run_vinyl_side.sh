#!/usr/bin/env bash
# Vinyl side jobs that run next to the Gemma overnight job (small models, < ~25 GB GPU each): J6 second local judge, then J11 (OLMo, Qwen). Own lock; log /root/atlas_logs/vinyl_side.log.
# Plans: docs/decisions.md 2026-11-05. Each step skipped if its output exists.
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
exec 8>/root/atlas_logs/.side.lock
flock -x 8
step() { local t=$(date +%s); "$@" || echo "FAILED: $*"; echo "   [$(( $(date +%s)-t )) s] $*" | cut -c1-160; }
echo "=== J6 $(date -u +%T)"
[[ -f results/overgeneralization/second_judge/traits.csv.gz ]] || step $PY -m extras.second_judge run
step $PY -m extras.second_judge analyze
for fam in olmo3-7b qwen3-8b; do for stage in instruct base; do
  [[ -f results/overgeneralization/skill_roleplay/$fam/$stage.csv.gz ]] && continue
  model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); cached=$(du -sm "$HUB/models--${model//\//--}" 2>/dev/null | cut -f1); [ "${cached:-0}" -gt 1000 ] || [ "$avail" -ge 17 ] || { echo "less than 17 GB free for $model"; continue; }
  echo "=== J11 $fam $stage $(date -u +%T)"
  step $PY -m extras.skill_roleplay run --family $fam --stage $stage
  free_model "$model"
done; done
echo "=== side done $(date -u +%T)"
