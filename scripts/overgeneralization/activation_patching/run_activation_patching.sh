#!/usr/bin/env bash
# Spec I (2026-11-04), runs on vinyl AFTER the in-character addendum (run_trait_atlas_incharacter.sh) and before atlas OLMo SFT/DPO and the final judge pass.
# Waits for the addendum script to end (it leaves /root/atlas_logs/HOLD_FOR_SPEC_I in place because /root/atlas_logs/SPEC_I_READY exists), then takes the exclusive GPU lock:
#   per model (OLMo instruct, then Qwen3-8B): Stage 0 tests (stops everything on a failure) -> Stage 1 (pilot, layer screen, floors.json) -> wait until the floors are committed
#   (the marble session commits and pushes floors.json, then writes floors.committed next to it) -> Stage 2 -> model freed with its blobs.
# Then analyze, lift the hold (atlas SFT/DPO and the final judge pass resume).
set -uo pipefail
cd "$(dirname "$0")/../../.."
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY=/opt/arena-env/bin/python
LOCK=results/overgeneralization/trait_atlas/.gpu.lock
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
touch /root/atlas_logs/HOLD_FOR_SPEC_I
echo "spec I waiting for the in-character addendum $(date -u +%T)"
while pgrep -f "[r]un_trait_atlas_incharacter.sh" >/dev/null; do sleep 30; done
(
  flock -x 9
  echo "=== spec I starts $(date -u +%T)"
  for fam in olmo3-7b qwen3-8b; do
    model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['instruct'])")
    echo "=== $fam stage 0 $(date -u +%T)"
    $PY -m extras.activation_patching stage0 --family $fam || { echo "STAGE 0 FAILED for $fam" | tee /root/atlas_logs/SPEC_I_STOP.txt; free_model "$model"; break; }
    echo "=== $fam stage 1 $(date -u +%T)"
    $PY -m extras.activation_patching stage1 --family $fam || { echo "STAGE 1 FAILED for $fam" | tee -a /root/atlas_logs/SPEC_I_STOP.txt; free_model "$model"; continue; }
    if [ ! -f results/overgeneralization/activation_patching/$fam/floors.json ]; then echo "$fam: no layer above the random range, Stage 2 skipped"; free_model "$model"; continue; fi
    echo "=== $fam waiting for committed floors $(date -u +%T)"
    for i in $(seq 1 120); do [ -f results/overgeneralization/activation_patching/$fam/floors.committed ] && break; sleep 30; done
    [ -f results/overgeneralization/activation_patching/$fam/floors.committed ] || { echo "floors never committed for $fam" | tee -a /root/atlas_logs/SPEC_I_STOP.txt; free_model "$model"; continue; }
    echo "=== $fam stage 2 $(date -u +%T)"
    $PY -m extras.activation_patching stage2 --family $fam || echo "STAGE 2 FAILED for $fam"
    free_model "$model"
  done
  $PY -m extras.activation_patching analyze
  echo "=== spec I ends $(date -u +%T)"
) 9>"$LOCK"
rm -f /root/atlas_logs/HOLD_FOR_SPEC_I /root/atlas_logs/SPEC_I_READY
echo "hold lifted, atlas continues $(date -u +%T)"
