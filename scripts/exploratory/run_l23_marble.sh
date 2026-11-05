#!/usr/bin/env bash
# Spec L, L2 + L3 on marble: Qwen3-8B, final first then base, after the bias queue (run_k_marble.sh). Log /tmp/claude-0/l23_marble.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_k_marble" >/dev/null || [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec L2/L3 Qwen; released when run_l23_marble.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for stage in instruct base; do
  [[ -f results/working_model_L/L23/qwen3-8b/${stage}_L3.csv.gz ]] && continue
  echo "=== qwen3-8b $stage $(date -u +%T)"
  $PY -m extras.working_model_l run23 --family qwen3-8b --stage $stage || echo "FAILED qwen $stage"
  case $stage in instruct) m=Qwen/Qwen3-8B;; *) m=Qwen/Qwen3-8B-Base;; esac
  free_model "$m"
done
echo "=== done $(date -u +%T)"
