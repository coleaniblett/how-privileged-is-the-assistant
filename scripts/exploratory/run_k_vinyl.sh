#!/usr/bin/env bash
# Spec K (BBQ) on vinyl: Gemma 4 31B instruct (cached) then base (bf16, batch 8); each model freed after use (62 GB). Log /root/atlas_logs/k_vinyl.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BL_BATCH=8
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
exec 9>results/trait_atlas/.gpu.lock
flock -x 9
for stage in instruct base; do
  [[ -f results/bias_leak/gemma4-31b/$stage.csv.gz ]] && continue
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); cached=$(du -sm "$HUB/models--google--gemma-4-31B-it" 2>/dev/null | cut -f1); [ "$stage" = instruct ] && [ "${cached:-0}" -gt 50000 ] || [ "$avail" -ge 70 ] || { echo "GEMMA $stage NOT STARTED: only ${avail} GB free" | tee /root/atlas_logs/K_GEMMA_STOP.txt; exit 4; }
  echo "=== gemma4-31b $stage $(date -u +%T)"
  $PY -m extras.bias_leak run --family gemma4-31b --stage $stage || echo "FAILED gemma $stage"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['$stage'])")"
done
echo "=== done $(date -u +%T)"
