#!/usr/bin/env bash
# Does 'role-play as' / 'respond only as' phrasing reduce the system-prompt refusal leak of 'never refuses' roles? OLMo sft and instruct.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for fr in rp only; do
  export RG_FRAME=$fr RG_OUT=results/olmo3-7b/role_gradient_$fr RG_REPORT=results/olmo3-7b/role_gradient_$fr.md
  for s in sft instruct; do $P -m extras.role_gradient run --stage $s; done
  $P -m extras.role_gradient analyze
done
