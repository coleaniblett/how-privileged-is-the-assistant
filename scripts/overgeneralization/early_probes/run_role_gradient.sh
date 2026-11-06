#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
for s in ${STAGES:-sft dpo instruct}; do $P -m extras.role_gradient run --stage $s; done
$P -m extras.role_gradient analyze
