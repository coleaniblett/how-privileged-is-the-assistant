#!/usr/bin/env bash
# Carryover (flat facts) on Gemma 4 E4B base vs instruct: balanced-frame holders, native-slot holders and the slot-ranking test. Contamination cannot be checked.
set -e
cd "$(dirname "$0")/../../.."
export CB_FACTS=1 CB_OUT=results/overgeneralization/early_probes/gemma-4-e4b/carryover CB_TOKENIZER=google/gemma-4-E4B-it
export CB_HOLDS=assistant,user,char,nat_assistant,nat_user,slot_system,slot_user,slot_assistant
P=/opt/arena-env/bin/python
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.carryover_balanced score --stage $m; done
$P -m extras.carryover_balanced analyze
mv results/overgeneralization/early_probes/olmo3-7b/carryover.md results/overgeneralization/early_probes/gemma-4-e4b/carryover.md
