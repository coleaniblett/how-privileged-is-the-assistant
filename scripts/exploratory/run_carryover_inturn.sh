#!/usr/bin/env bash
# Position control for the slot ranking: fact stated by the Assistant in the same turn right before the character answers.
# OLMo 4 stages, then Gemma base/instruct (reports are written under results/overgeneralization/early_probes/olmo3-7b by the module and the Gemma one is moved).
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
export CB_FACTS=1 CB_HOLDS=slot_user,slot_asst_inturn,slot_assistant
export CB_OUT=results/overgeneralization/early_probes/olmo3-7b/carryover_inturn
for s in base sft dpo instruct; do $P -m extras.carryover_balanced score --stage $s; done
$P -m extras.carryover_balanced analyze
export CB_OUT=results/overgeneralization/early_probes/gemma-4-e4b/carryover_inturn_gemma CB_TOKENIZER=google/gemma-4-E4B-it
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.carryover_balanced score --stage $m; done
$P -m extras.carryover_balanced analyze
mv results/overgeneralization/early_probes/olmo3-7b/carryover_inturn_gemma.md results/overgeneralization/early_probes/gemma-4-e4b/carryover_inturn.md
