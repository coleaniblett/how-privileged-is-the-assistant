#!/usr/bin/env bash
# Spec J5 (plan: docs/decisions.md 2026-11-05) on vinyl: more base samples for OLMo and Qwen light roles, detectors, local judges, pooled addendum analysis. Log /root/atlas_logs/j5.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
exec 9>results/overgeneralization/trait_atlas/.gpu.lock
flock -x 9
step() { local t=$(date +%s); "$@" || echo "FAILED: $*"; echo "   [$(( $(date +%s)-t )) s] $*" | cut -c1-200; }
for fam in olmo3-7b qwen3-8b; do
  [[ -f results/overgeneralization/trait_atlas_more_base/$fam/base_replies.jsonl.gz ]] || step $PY -m extras.trait_atlas_more_base gen --family $fam --batch-size 32
  free_model "$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['base'])")"
done
step $PY -m extras.trait_atlas_more_base detect
step $PY -m extras.trait_atlas_more_base judge --batch-size 16
step $PY -m extras.trait_atlas_more_base analyze
echo "=== done $(date -u +%T)"
