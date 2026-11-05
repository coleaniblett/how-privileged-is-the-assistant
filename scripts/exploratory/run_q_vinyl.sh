#!/usr/bin/env bash
# Spec Q on vinyl: after spec N and spec O's Gemma arm (shared GPU lock queue): Gemma 4 31B final generation (batch 24; Gemma BASE is dropped because Q projects over 4 GPU hours, see docs/decisions.md),
# then the local judge over every replies file (waits for the marble replies to arrive). Log /root/atlas_logs/q_vinyl.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_n_vinyl" >/dev/null || pgrep -f "[r]un_o_vinyl" >/dev/null; do sleep 30; done
exec 9>results/trait_atlas/.gpu.lock
flock -x 9
mkdir -p results/role_says/gemma4-31b
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "NOT STARTED: only ${avail} GB free" | tee /root/atlas_logs/Q_STOP.txt; exit 4; }
[[ -f results/role_says/gemma4-31b/instruct_replies.csv.gz ]] || $PY -m extras.role_says gen --family gemma4-31b --stage instruct || echo "FAILED gemma instruct"
echo "=== generation done $(date -u +%T)"
# the judge is Gemma 4 31B-it, which is already cached after the generation; wait for the OLMo/Qwen replies from marble
until [ -f results/role_says/MARBLE_REPLIES_DONE ]; do sleep 30; done
$PY -m extras.role_says judge || echo "FAILED judge"
free_model "$($PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['instruct'])")"
touch results/role_says/Q_JUDGED_DONE
echo "=== done $(date -u +%T)"
