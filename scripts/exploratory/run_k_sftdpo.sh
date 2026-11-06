#!/usr/bin/env bash
# Spec K, OLMo SFT and DPO (descriptive timing only), after spec L on marble. Log /tmp/claude-0/k_sftdpo.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_l14_marble" >/dev/null || [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec K OLMo SFT/DPO; released when run_k_sftdpo.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for stage in sft dpo; do
  [[ -f results/overgeneralization/bias_leak/olmo3-7b/$stage.csv.gz ]] && continue
  echo "=== olmo3-7b $stage $(date -u +%T)"
  $PY -m extras.bias_leak run --family olmo3-7b --stage $stage || echo "FAILED $stage"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['olmo3-7b']['stages']['$stage'])")"
done
echo "=== done $(date -u +%T)"
