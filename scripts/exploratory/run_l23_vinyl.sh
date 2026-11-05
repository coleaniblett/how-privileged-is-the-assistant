#!/usr/bin/env bash
# Spec L, L2 + L3 on vinyl: OLMo 3 7B and Gemma 4 E4B, final first then base; waits for the bias test's Gemma job (shared GPU lock). Log /root/atlas_logs/l23_vinyl.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
exec 9>results/trait_atlas/.gpu.lock
flock -x 9
for fam in olmo3-7b gemma-e4b; do for stage in instruct base; do
  [[ -f results/working_model_L/L23/$fam/${stage}_L3.csv.gz ]] && continue
  echo "=== $fam $stage $(date -u +%T)"
  $PY -m extras.working_model_l run23 --family $fam --stage $stage || echo "FAILED $fam $stage"
  case "$fam $stage" in "olmo3-7b instruct") m=allenai/Olmo-3-7B-Instruct;; "olmo3-7b base") m=allenai/Olmo-3-1025-7B;; "gemma-e4b instruct") m=google/gemma-4-E4B-it;; *) m=google/gemma-4-E4B;; esac
  free_model "$m"
done; done
echo "=== done $(date -u +%T)"
