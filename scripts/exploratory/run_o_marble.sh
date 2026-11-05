#!/usr/bin/env bash
# Spec O on marble (OLMo, Qwen): sanity check, arm 2 (persona prefill, base and final, scoring + continuations), arm 3 (steering, final), analysis.
# Stops with /tmp/claude-0/O_STOP.txt if the sanity check fails or the elapsed time passes 2 h. Log /tmp/claude-0/o_marble.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
D=data/belief_induction_prefill.jsonl
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec O; released when run_o_marble.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
T0=$(date +%s)
step() { local t=$(date +%s); "$@" || echo "FAILED: $*"; echo "   [$(( $(date +%s)-t )) s] $*" | cut -c1-170; if [ $(( $(date +%s)-T0 )) -gt 7200 ]; then echo "O over 2 h" | tee /tmp/claude-0/O_STOP.txt; exit 9; fi; }
echo "=== sanity $(date -u +%T)"
step $PY -m extras.belief_induction sanity
mx=$($PY -c "import json;print(json.load(open('results/belief_induction/sanity.json'))['max_abs_diff'])")
echo "sanity max abs diff $mx"
$PY -c "import sys; sys.exit(0 if float('$mx') < 0.05 else 1)" || { echo "SANITY FAILED: max abs diff $mx" | tee /tmp/claude-0/O_STOP.txt; exit 8; }
for fam in olmo3-7b qwen3-8b; do
  for stage in instruct base; do
    O=results/belief_induction/arm2/$fam
    mkdir -p $O/persona_carryover
    echo "=== arm 2 $fam $stage $(date -u +%T)"
    [[ -f $O/persona_carryover/$stage.csv ]] || step $PY -m measures.overgeneralization.persona_carryover score --family $fam --model $stage --stage $stage --dataset $D --out $O/persona_carryover
    [[ -f $O/${stage}_continuations.csv ]] || step $PY -m extras.belief_induction continuations --family $fam --stage $stage
    m=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
    [ "$stage" = base ] && free_model "$m"
  done
  step $PY -m measures.overgeneralization.persona_carryover analyze --family $fam --verdict --dataset $D --scores results/belief_induction/arm2/$fam/persona_carryover --out-dir results/belief_induction/arm2/$fam
done
for fam in olmo3-7b qwen3-8b; do
  echo "=== arm 3 $fam $(date -u +%T)"
  [[ -f results/belief_induction/arm3/$fam/meta.json ]] || step $PY -m extras.belief_induction steer --family $fam
  m=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['instruct'])"); free_model "$m"
done
step $PY -m extras.belief_induction analyze
echo "=== done $(date -u +%T) elapsed $(( $(date +%s)-T0 )) s"
