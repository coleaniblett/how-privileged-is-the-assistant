#!/usr/bin/env bash
# Spec J2 (plan: docs/decisions.md 2026-11-05): Llama 3.1 8B and Apertus 8B 2509, instruct then base, one checkpoint at a time:
# belief test, prefill generations (200 + 400 prefills), Assistant text in the user turn (900 exchanges), spec H2 and H3; then the analyses.
# No judging here. Log: /tmp/claude-0/j2.log. Each step is skipped if its output exists, so a rerun resumes.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
DATA=data/persona_carryover_confirmatory.jsonl
[[ "$(sha256sum $DATA | cut -c1-8)" == e319184f ]] || { echo "carryover set changed"; exit 2; }
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=J2 new families; released when run_j2_new_families.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
step() { local t=$(date +%s); "$@" || echo "FAILED: $*"; echo "   [$(( $(date +%s)-t )) s] $*" | cut -c1-200; }
for fam in llama3.1-8b apertus-8b; do
  for stage in instruct base; do
    model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
    avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 25 ] || { echo "less than 25 GB free for $model"; exit 3; }
    echo "=== $fam $stage ($model) $(date -u +%T)"
    CO=results/overgeneralization/persona_carryover/$fam/confirmatory/persona_carryover; mkdir -p $CO
    [[ -f $CO/$stage.csv ]] || step $PY -m measures.overgeneralization.persona_carryover score --family $fam --model $stage --stage $stage --dataset $DATA --out $CO
    [[ -f results/capstone/$fam/core/generations_incomplete/$stage.jsonl ]] || step $PY -m core.generate --family $fam --run capstone --model $stage --prefill data/prefill_incomplete_200.jsonl --save-ids --out results/capstone/$fam/core/generations_incomplete
    [[ -f results/capstone/$fam/core/generations_incomplete_exchanges/$stage.jsonl ]] || step $PY -m core.generate --family $fam --run capstone --model $stage --prefill data/prefill_incomplete_exchanges.jsonl --save-ids --out results/capstone/$fam/core/generations_incomplete_exchanges
    step $PY -m extras.exchange_reply_in_user score --family $fam --model $stage --offload --gpu-gib 20
    [[ -f results/overgeneralization/mechanism_checks/h2/$fam/$stage.jsonl.gz ]] || step $PY -m extras.mechanism_checks h2 --family $fam --stage $stage
    [[ -f results/overgeneralization/mechanism_checks/h3/$fam/$stage.jsonl.gz ]] || step $PY -m extras.mechanism_checks h3 --family $fam --stage $stage
    free_model "$model"
  done
  step $PY -m measures.overgeneralization.persona_carryover analyze --family $fam --verdict --dataset $DATA --scores results/overgeneralization/persona_carryover/$fam/confirmatory/persona_carryover --out-dir results/overgeneralization/persona_carryover/$fam/confirmatory
  step $PY -m extras.exchange_reply_in_user analyze --family $fam
  echo "=== $fam generations done $(date -u +%T)"
done
echo "=== done $(date -u +%T)"
