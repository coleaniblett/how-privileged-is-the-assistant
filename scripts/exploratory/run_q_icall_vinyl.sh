#!/usr/bin/env bash
# Spec Q follow-up on vinyl (approved 2026-11-05 23:40 UTC): in-character judge on ALL final role replies (3 families). Takes the shared GPU lock. Log /root/atlas_logs/q_icall_vinyl.log
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
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "NOT STARTED: only ${avail} GB free" | tee /root/atlas_logs/Q_ICALL_STOP.txt; exit 4; }
$PY -m extras.role_says judge_ic_all || echo "FAILED icall"
free_model "$($PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['instruct'])")"
touch results/role_says/Q_ICALL_DONE
echo "=== done $(date -u +%T)"
