#!/usr/bin/env bash
# R3 on vinyl: Gemma 31B persona prefill, wording 2 (final), with continuations; wording 1 exists in results/belief_induction/arm2/gemma4-31b. Same opening as arm 2 (empty thought channel).
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; R=results/overnight_2026-11-05/R3; W2=data/belief_induction_prefill_w2.jsonl
stage=${1:-instruct}; O=$R/w2/arm2/gemma4-31b; mkdir -p $O/persona_carryover
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || [ -d "$HOME/.cache/huggingface/hub/models--google--gemma-4-31B-it" -a $stage = instruct ] || { echo "only ${avail} GB free"; exit 4; }
[[ -f $O/persona_carryover/$stage.csv ]] || $PY -m measures.overgeneralization.persona_carryover score --family gemma4-31b --model $stage --stage $stage --dataset $W2 --out $O/persona_carryover --assistant-opening --gpu-gib 90 || exit 1
[ $stage = instruct ] && { [[ -f $O/${stage}_continuations.csv ]] || BI_DATA=$W2 BI_OUT=$R/w2 $PY -m extras.belief_induction continuations --family gemma4-31b --stage $stage || exit 1; }
if [ $stage = base ]; then
  $PY -m measures.overgeneralization.persona_carryover analyze --family gemma4-31b --verdict --dataset $W2 --scores $O/persona_carryover --out-dir $O --assistant-opening
  source scripts/exploratory/lib_r.sh; free_model "$($PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['base'])")"; touch $R/GEMMA_W2_DONE
fi
