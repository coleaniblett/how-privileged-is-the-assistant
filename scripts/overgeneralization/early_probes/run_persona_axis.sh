#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
for s in ${STAGES:-base sft dpo instruct}; do $P -m extras.persona_axis fit --stage $s; done
$P -m extras.persona_axis intervene --stage instruct
$P -m extras.persona_axis analyze
