#!/usr/bin/env bash
# Is the bare-role collapse specific to the system-prompt slot? Persona in the user turn instead (bare and full forms), OLMo 4 stages and Gemma base/instruct.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
export RR_SLOT=user
for form in plain full; do
  export RR_FORM=$form RR_OUT=results/overgeneralization/early_probes/olmo3-7b/role_register_userslot_$form RR_REPORT=results/overgeneralization/early_probes/olmo3-7b/role_register_userslot_$form.md
  for s in base sft dpo instruct; do $P -m extras.role_register run --stage $s --axis-stage instruct; done
  $P -m extras.role_register analyze
done
for form in plain full; do
  export RR_FORM=$form RR_OUT=results/overgeneralization/early_probes/other/role_register_gemma_userslot_$form RR_REPORT=results/overgeneralization/early_probes/other/role_register_gemma_userslot_$form.md PA_OUT=results/overgeneralization/early_probes/gemma-4-e4b/persona_axis
  for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.role_register run --stage $m --tokenizer google/gemma-4-E4B-it --axis-stage gemma-4-E4B-it; done
  $P -m extras.role_register analyze
done
