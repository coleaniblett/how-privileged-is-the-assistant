#!/usr/bin/env bash
# Spec J3 (plan: docs/decisions.md 2026-11-05): refusal direction, final models first then base. Waits for J2 (run_j2_new_families.sh) and the GPU. Log /tmp/claude-0/j3.log.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_j2_new_families" >/dev/null || [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 30; done
echo "owner=main-session purpose=J3 refusal direction; released when run_j3.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for stage in instruct base; do for fam in olmo3-7b qwen3-8b; do
  [[ -f results/overgeneralization/refusal_direction/$fam/$stage/gens.csv.gz ]] || { echo "=== $fam $stage $(date -u +%T)"; $PY -m extras.refusal_direction run --family $fam --stage $stage || echo "FAILED $fam $stage"; free_model "$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")"; }
done; done
$PY -m extras.refusal_direction analyze
echo "=== done $(date -u +%T)"
