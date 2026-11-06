#!/usr/bin/env bash
# Spec L, L1 then L4 on vinyl for Gemma 4 31B (final first, then base; weights downloaded and freed one at a time), after the L2/L3 vinyl job. Log /root/atlas_logs/l14_vinyl.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WML_BS=24
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
exec 9>results/overgeneralization/trait_atlas/.gpu.lock
flock -x 9
for stage in instruct base; do
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "GEMMA $stage NOT STARTED: only ${avail} GB free" | tee /root/atlas_logs/L_GEMMA_STOP.txt; exit 4; }
  for test in 1 4; do
    [[ -f results/overgeneralization/working_model_L/L$test/gemma4-31b/$stage.csv.gz ]] && continue
    echo "=== L$test gemma4-31b $stage $(date -u +%T)"
    $PY -m extras.working_model_l run$test --family gemma4-31b --stage $stage || echo "FAILED L$test gemma $stage"
  done
  m=$($PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['$stage'])"); free_model "$m"
done
echo "=== done $(date -u +%T)"
