#!/usr/bin/env bash
# Spec G run order on vinyl for early readings (no design change): (1) after OLMo base/instruct and Qwen base/instruct are generated: detect, judge (timed), hand-check
# sheet, analyze (judge results provisional); (2) the Gemma carryover rerun with the thought channel; (3) the remaining atlas checkpoints (Gemma instruct and base at batch 48,
# OLMo SFT, DPO); (4) detect, judge for the new checkpoints (incremental), final analyze. Run from the worktree; never touches /root/g4nla.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1
PY=/opt/arena-env/bin/python
LOG=/root/atlas_logs
( while sleep 30; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> $LOG/gpumem.log; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
while ! grep -q base-done $LOG/after_base2.log 2>/dev/null; do sleep 20; done
while pgrep -f "[r]un_trait_atlas_queue" >/dev/null || pgrep -f "[e]xtras.trait_atlas generate" >/dev/null; do sleep 30; done
echo "=== phase 1: detect + judge $(date -u +%T)"
$PY -m extras.trait_atlas detect
t0=$(date +%s)
$PY -m extras.trait_atlas judge --batch-size 16
t1=$(date +%s)
n=$($PY - <<PY2
import glob, pandas as pd
print(sum(len(pd.read_csv(p)) for p in glob.glob("results/overgeneralization/trait_atlas/*/*_judged*.csv.gz")))
PY2
)
echo "JUDGE PASS 1: $n judgements in $((t1-t0)) s = $(( n * 60 / (t1-t0+1) )) per minute"
$PY -m extras.trait_atlas handcheck
$PY -m extras.trait_atlas analyze
echo "=== phase 1 done $(date -u +%T)"
echo "=== phase 2: Gemma carryover rerun with thought channel $(date -u +%T)"
FROZEN_SHA256=e319184f4dd09f6d705d2bdd8badc2575919b0d4a0913c14e84ef83708b23344 bash scripts/exploratory/run_carryover_thought_channel_gemma.sh
echo "=== phase 3: remaining atlas checkpoints $(date -u +%T)"
rmdir results/overgeneralization/trait_atlas/.claims/gemma4-31b_instruct results/overgeneralization/trait_atlas/.claims/gemma4-31b_base results/overgeneralization/trait_atlas/.claims/olmo3-7b_sft results/overgeneralization/trait_atlas/.claims/olmo3-7b_dpo 2>/dev/null
WORKER=w3 CACHE=/dev/shm/hf_w3 bash scripts/exploratory/run_trait_atlas_queue2.sh
echo "=== phase 4: final detect, judge, analyze $(date -u +%T)"
$PY -m extras.trait_atlas detect
$PY -m extras.trait_atlas judge --batch-size 16
$PY -m extras.trait_atlas analyze
echo "=== all done $(date -u +%T)"
