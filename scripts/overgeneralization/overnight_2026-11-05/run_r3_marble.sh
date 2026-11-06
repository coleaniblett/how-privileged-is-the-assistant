#!/usr/bin/env bash
# Spec R3 on marble: persona prefill, wording 1 (Llama, Apertus) and wording 2 (all of OLMo, Qwen, Llama, Apertus), base and final, plus 30-per-condition continuations (final) for the in-character judge.
# Output under results/overgeneralization/overnight_2026-11-05/R3/{w1,w2}/arm2/<family>/. Log /tmp/claude-0/r3_marble.log
set -uo pipefail
cd /root/hpa-main-wt
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
R=results/overgeneralization/overnight_2026-11-05/R3
W1=data/belief_induction_prefill.jsonl; W2=data/belief_induction_prefill_w2.jsonl
gpu_wait main-session "spec R3 marble"; trap gpu_release EXIT
run() { local w=$1 D=$2 fam=$3 stage=$4
  local O=$R/$w/arm2/$fam; mkdir -p $O/persona_carryover
  [[ -f $O/persona_carryover/$stage.csv ]] || retry $PY -m measures.overgeneralization.persona_carryover score --family $fam --model $stage --stage $stage --dataset $D --out $O/persona_carryover
  if [ "$stage" = instruct ]; then [[ -f $O/${stage}_continuations.csv ]] || BI_DATA=$D BI_OUT=$R/$w retry $PY -m extras.belief_induction continuations --family $fam --stage $stage; fi; }
for fam in olmo3-7b qwen3-8b llama3.1-8b apertus-8b; do
  for stage in instruct base; do
    echo "=== $fam $stage $(date -u +%T)"
    [ "$fam" = llama3.1-8b -o "$fam" = apertus-8b ] && run w1 $W1 $fam $stage
    run w2 $W2 $fam $stage
    [ "$stage" = base ] && free_if_tight "$(stage_model $fam base)" 45
  done
  for w in w1 w2; do D=$W1; [ $w = w2 ] && D=$W2
    [ "$w" = w1 -a \( "$fam" = olmo3-7b -o "$fam" = qwen3-8b \) ] && continue
    O=$R/$w/arm2/$fam; retry $PY -m measures.overgeneralization.persona_carryover analyze --family $fam --verdict --dataset $D --scores $O/persona_carryover --out-dir $O
  done
  [ "$fam" = llama3.1-8b -o "$fam" = apertus-8b ] && free_if_tight "$(stage_model $fam instruct)" 40
done
echo "=== R3 marble done $(date -u +%T)"; touch $R/MARBLE_DONE
push_paths "results: spec R3 marble (prefill belief test, wording 1 and 2, Llama/Apertus/OLMo/Qwen; raw scores and continuations, no analysis yet)" $R
