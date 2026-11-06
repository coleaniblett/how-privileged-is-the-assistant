#!/usr/bin/env bash
# Vinyl (plan: docs/decisions.md 2026-11-05 "Vinyl plan"): Gemma 4 31B exchange prefills (no judging), user-slot scoring, spec H2/H3. Instruct first (cached), then base.
# Exclusive GPU lock like the atlas workers. Log: /root/atlas_logs/gemma_overnight.log. Each step is skipped if its output exists.
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
exec 9>results/overgeneralization/trait_atlas/.gpu.lock
flock -x 9
FAM=gemma4-31b
step() { local t=$(date +%s); "$@" || echo "FAILED: $*"; echo "   [$(( $(date +%s)-t )) s] $*" | cut -c1-200; }
for stage in instruct base; do
  model=$($PY -c "from core import config; print(config.FAMILIES['$FAM']['stages']['$stage'])")
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || [ "$stage" = instruct ] || { echo "less than 70 GB free for $model"; exit 3; }
  echo "=== $FAM $stage $(date -u +%T)"
  out=results/capstone/$FAM/core/generations_incomplete_exchanges
  [[ -f $out/$stage.jsonl ]] || step $PY -m extras.batched_generate --family $FAM --run capstone --model $stage --prefill data/prefill_incomplete_exchanges.jsonl --out $out
  [[ -f results/capstone/$FAM/$stage/core/exchanges_reply_in_user/scores.csv ]] || step $PY -m extras.exchange_reply_in_user score --family $FAM --model $stage --offload --gpu-gib 30
  [[ -f results/overgeneralization/mechanism_checks/h2/$FAM/$stage.jsonl.gz ]] || step $PY -m extras.mechanism_checks h2 --family $FAM --stage $stage
  [[ -f results/overgeneralization/mechanism_checks/h3/$FAM/$stage.jsonl.gz ]] || step $PY -m extras.mechanism_checks h3 --family $FAM --stage $stage
  free_model "$model"
done
step $PY -m extras.exchange_reply_in_user analyze --family $FAM
echo "=== done $(date -u +%T)"
