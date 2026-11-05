#!/usr/bin/env bash
# Spec J4 (plan: docs/decisions.md 2026-11-05) on marble after J8: screen both models, freeze + commit the roles, then Stage 2 for both, then analyze. Log /tmp/claude-0/j4.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_j8_lowdose" >/dev/null || [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=J4 patching new roles; released when run_j4.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for fam in olmo3-7b qwen3-8b; do
  [[ -f results/activation_patching_j4/$fam/screen_roles.csv ]] || $PY -m extras.activation_patching_j4 screen --family $fam || echo "FAILED screen $fam"
done
$PY -m extras.activation_patching_j4 freeze
git add results/activation_patching_j4/*/frozen_materials.json results/activation_patching_j4/*/screen_roles.csv
git commit -q -m "results: spec J4 frozen roles per model (committed before Stage 2)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && { git pull -q --rebase; git push -q origin HEAD:main; }
for fam in olmo3-7b qwen3-8b; do
  [[ -f results/activation_patching_j4/$fam/stage2_gens.csv.gz ]] || $PY -m extras.activation_patching_j4 stage2 --family $fam || echo "FAILED stage2 $fam"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['instruct'])")"
done
$PY -m extras.activation_patching_j4 analyze
echo "=== done $(date -u +%T)"
