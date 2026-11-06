#!/usr/bin/env bash
# Spec G addendum (POST-HOC, 2026-11-04): in-character judge pass. Runs on vinyl right after atlas Gemma base is generated, BEFORE spec I and before atlas OLMo SFT/DPO.
# Nothing running is restarted: HOLD_FOR_SPEC_I keeps the queue's OLMo claims and the final judge pass waiting (Gemma is not held); the exclusive GPU lock is taken after the Gemma worker releases it.
# Steps: sample rate (200 prompts) -> projection (stops with /root/atlas_logs/INCHARACTER_STOP.txt if > 60 min) -> judge -> handcheck -> analyze.
# At the end the hold is lifted unless /root/atlas_logs/SPEC_I_READY exists (then spec I takes the slot).
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY=/opt/arena-env/bin/python
LOCK=results/overgeneralization/trait_atlas/.gpu.lock
touch /root/atlas_logs/HOLD_FOR_SPEC_I
echo "hold set $(date -u +%T); waiting for atlas Gemma base"
until [[ -f results/overgeneralization/trait_atlas/gemma4-31b/base_replies.jsonl.gz ]]; do sleep 60; done
sleep 60
(
  flock -x 9
  echo "=== in-character pass starts $(date -u +%T)"
  $PY -m extras.trait_atlas_incharacter judge --sample 200 2>&1 | tee /root/atlas_logs/incharacter_sample.log | grep -E "SAMPLE|Error|error"
  rate=$(grep -oE "= [0-9]+ per minute" /root/atlas_logs/incharacter_sample.log | tail -1 | grep -oE "[0-9]+")
  n=$($PY - <<'P'
import glob, os, pandas as pd
from extras import trait_atlas as ta, trait_atlas_incharacter as ic
n = 0
for p in glob.glob("results/overgeneralization/trait_atlas/*/*_replies.jsonl.gz"):
    fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
    if not os.path.exists(f"results/overgeneralization/trait_atlas_incharacter/atlas_{fam}_{stage}.csv.gz"):
        n += 2880
for fam, stages in ic.LEAK_FAMS.items():
    for s in stages:
        if os.path.exists(f"{ic.LEAK}/{fam}/{s}.jsonl") and not os.path.exists(f"results/overgeneralization/trait_atlas_incharacter/leak_{fam}_{s}.csv.gz"):
            n += 1152
print(n)
P
)
  proj=$(( n / ${rate:-1} + 10 ))
  echo "RATE ${rate:-?} per minute, ${n} judgements, projected ~${proj} min incl. ~10 min load/download"
  echo "$(date -u +%T) rate ${rate:-?}/min, n=${n}, projected ${proj} min" > /root/atlas_logs/incharacter_projection.txt
  if [ "$proj" -gt 60 ]; then
    echo "projected ${proj} min > 60: stopping for a decision" | tee /root/atlas_logs/INCHARACTER_STOP.txt
  else
    $PY -m extras.trait_atlas_incharacter judge && $PY -m extras.trait_atlas_incharacter handcheck && $PY -m extras.trait_atlas_incharacter analyze
  fi
  echo "=== in-character pass ends $(date -u +%T)"
) 9>"$LOCK"
if [ -e /root/atlas_logs/SPEC_I_READY ]; then echo "spec I ready: hold stays"; else rm -f /root/atlas_logs/HOLD_FOR_SPEC_I; echo "spec I not ready: hold lifted, atlas continues"; fi
