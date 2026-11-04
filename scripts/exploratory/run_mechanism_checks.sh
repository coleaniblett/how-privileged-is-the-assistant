#!/usr/bin/env bash
# Spec H (2026-11-04): H1, H2, H3 per checkpoint (one model load per checkpoint, one GPU job at a time), then analyze. Outputs results/mechanism_checks/.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
export GENS_DIR=/root/how-privileged-is-the-assistant/results/dominance_matrix/gens MC_RESULTS_SRC=/root/how-privileged-is-the-assistant/results
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 30; done
echo "owner=main-session purpose=spec H mechanism checks; released when run_mechanism_checks.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
t() { local s=$(date +%s); "$@"; echo "   [$(( $(date +%s)-s )) s] $*"; }
# lm-family  config-family  stage  hf-id(resolved)  parts
for spec in "gemma gemma4-e4b base - h1" "gemma gemma4-e4b instruct - h1" \
            "olmo olmo3-7b base - h1 h2 h3" "olmo olmo3-7b instruct - h1 h2 h3" "olmo olmo3-7b sft - h2" "olmo olmo3-7b dpo - h2" \
            "qwen qwen3-8b base - h1 h2 h3" "qwen qwen3-8b instruct - h1 h2 h3"; do
  set -- $spec; lmf=$1; cf=$2; stage=$3; shift 4
  echo "=== $cf $stage $(date -u +%T) parts: $*"
  for p in "$@"; do
    if [ $p = h1 ]; then fam=$lmf; else fam=$cf; fi
    t $PY -m extras.mechanism_checks $p --family $fam --stage $stage || echo "FAILED $p $fam $stage"
  done
  model=$($PY -c "
from extras import leakage_matrix as lm
from core.score import resolve
print(resolve(lm.FAMILIES['$lmf']['stages']['$stage'])[0])" 2>/dev/null | tail -1)
  [ -n "$model" ] && [[ "$model" == */* ]] && [[ "$model" != /* ]] && free_model "$model"
done
$PY -m extras.mechanism_checks analyze
echo "=== done $(date -u +%T)"
