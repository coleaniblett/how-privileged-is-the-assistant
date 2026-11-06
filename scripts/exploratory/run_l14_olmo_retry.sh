#!/usr/bin/env bash
# Spec L: OLMo L1 and L4 retry (the first run hit CUDA OOM at batch 96 on OLMo's long role prompts; batch 32), then the spec K OLMo SFT/DPO bias runs. Log /tmp/claude-0/l14_retry.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WML_BS=32
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec L OLMo L1/L4 retry then K SFT/DPO; released when run_l14_olmo_retry.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for stage in instruct base; do for test in 1 4; do
  [[ -f results/overgeneralization/working_model_L/L$test/olmo3-7b/$stage.csv.gz ]] && continue
  echo "=== L$test olmo3-7b $stage $(date -u +%T)"
  $PY -m extras.working_model_l run$test --family olmo3-7b --stage $stage || echo "FAILED L$test olmo $stage"
done; free_model "$($PY -c "from core import config; print(config.FAMILIES['olmo3-7b']['stages']['$stage'])")"; done
unset WML_BS
for stage in sft dpo; do
  [[ -f results/overgeneralization/bias_leak/olmo3-7b/$stage.csv.gz ]] && continue
  echo "=== K olmo3-7b $stage $(date -u +%T)"
  $PY -m extras.bias_leak run --family olmo3-7b --stage $stage || echo "FAILED K $stage"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['olmo3-7b']['stages']['$stage'])")"
done
echo "=== done $(date -u +%T)"
