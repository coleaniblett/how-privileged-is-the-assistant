#!/usr/bin/env bash
# Spec M: judge-free atlas on one family (FAMILY=llama3.1-8b on marble, apertus-8b on vinyl), final then base, then the detectors. Log: first argument or stdout.
set -uo pipefail
FAM=${FAMILY:?set FAMILY}
cd "${REPO:-$(dirname "$0")/../..}"
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
if [ -n "${GPULOCK:-}" ]; then exec 9>"$GPULOCK"; flock -x 9; else
  while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
  echo "owner=main-session purpose=spec M $FAM; released when run_m_breadth.sh ends" > /tmp/claude-0/gpu.lock; trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
fi
for stage in instruct base; do
  [[ -f results/trait_atlas_breadth/$FAM/${stage}_replies.jsonl.gz ]] && continue
  echo "=== $FAM $stage $(date -u +%T)"
  $PY -m extras.trait_atlas_breadth gen --family $FAM --stage $stage || echo "FAILED $FAM $stage"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['$FAM']['stages']['$stage'])")"
done
$PY -m extras.trait_atlas_breadth detect
echo "=== done $(date -u +%T)"
