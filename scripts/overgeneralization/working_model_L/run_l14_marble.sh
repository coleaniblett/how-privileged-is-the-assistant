#!/usr/bin/env bash
# Spec L, L1 then L4 on marble for OLMo 3 7B and Qwen3-8B (final first), after the L2/L3 Qwen job. Log /tmp/claude-0/l14_marble.log
set -uo pipefail
cd "$(dirname "$0")/../../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_l23_marble" >/dev/null || pgrep -f "[r]un_k_marble" >/dev/null || [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec L1/L4 marble; released when run_l14_marble.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for test in 1 4; do for fam in olmo3-7b qwen3-8b; do for stage in instruct base; do
  [[ -f results/overgeneralization/working_model_L/L$test/$fam/$stage.csv.gz ]] && continue
  echo "=== L$test $fam $stage $(date -u +%T)"
  $PY -m extras.working_model_l run$test --family $fam --stage $stage || echo "FAILED L$test $fam $stage"
  m=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])"); free_model "$m"
done; done; done
echo "=== done $(date -u +%T)"
