#!/usr/bin/env bash
# Which system-prompt framings keep OLMo instruct in the Assistant register? (36 bare roles, fixed instruct persona axis)
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for form in rp only adopt; do
  export RR_FORM=$form RR_OUT=results/olmo3-7b/role_register_sys_$form RR_REPORT=results/olmo3-7b/role_register_sys_$form.md
  for s in sft instruct; do $P -m extras.role_register run --stage $s --axis-stage instruct; done
  $P -m extras.role_register analyze
done
