#!/usr/bin/env bash
# Spec O arm 2 on Gemma 4 31B (vinyl), after spec N (run_n_vinyl.sh) without disturbing it: instruct then base (download, score, continuations, free), then the analysis. Log /root/atlas_logs/o_vinyl.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
D=data/belief_induction_prefill.jsonl
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_n_vinyl" >/dev/null; do sleep 30; done
exec 9>results/trait_atlas/.gpu.lock
flock -x 9
fam=gemma4-31b; O=results/belief_induction/arm2/$fam
mkdir -p $O/persona_carryover
for stage in instruct base; do
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "NOT STARTED $stage: only ${avail} GB free" | tee /root/atlas_logs/O_GEMMA_STOP.txt; exit 4; }
  echo "=== arm 2 $fam $stage $(date -u +%T)"
  [[ -f $O/persona_carryover/$stage.csv ]] || $PY -m measures.overgeneralization.persona_carryover score --family $fam --model $stage --stage $stage --dataset $D --out $O/persona_carryover --assistant-opening --gpu-gib 90 || echo "FAILED score $stage"
  [[ -f $O/${stage}_continuations.csv ]] || $PY -m extras.belief_induction continuations --family $fam --stage $stage || echo "FAILED continuations $stage"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")"
done
$PY -m measures.overgeneralization.persona_carryover analyze --family $fam --verdict --dataset $D --scores $O/persona_carryover --out-dir $O --assistant-opening
echo "=== done $(date -u +%T)"
