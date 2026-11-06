#!/usr/bin/env bash
# Spec K (BBQ) on marble: Qwen3-8B, OLMo 3 7B, then Llama 3.1 8B and Apertus 8B; base then final; model freed after each checkpoint. Log /tmp/claude-0/k_marble.log
set -uo pipefail
cd "$(dirname "$0")/../../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec K BBQ marble; released when run_k_marble.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for fam in qwen3-8b olmo3-7b llama3.1-8b apertus-8b; do for stage in base instruct; do
  [[ -f results/overgeneralization/bias_leak/$fam/$stage.csv.gz ]] && continue
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 25 ] || { echo "less than 25 GB free for $fam $stage"; continue; }
  echo "=== $fam $stage $(date -u +%T)"
  $PY -m extras.bias_leak run --family $fam --stage $stage || echo "FAILED $fam $stage"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")"
done; done
echo "=== done $(date -u +%T)"
