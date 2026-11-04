#!/usr/bin/env bash
# What makes a role 'thick'? Same 36 roles with a bare description, a first name, or a full name + "Stay in character". OLMo sft/dpo/instruct, fixed instruct axis.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for form in named full; do
  export RR_FORM=$form RR_OUT=results/olmo3-7b/role_register_$form RR_REPORT=results/olmo3-7b/role_register_$form.md
  for s in base sft dpo instruct; do $P -m extras.role_register run --stage $s --axis-stage instruct; done
  $P -m extras.role_register analyze
done
