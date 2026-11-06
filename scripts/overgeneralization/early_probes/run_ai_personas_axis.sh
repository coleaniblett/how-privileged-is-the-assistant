#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
for s in ${STAGES:-base sft dpo instruct}; do $P -m extras.ai_personas_axis run --stage $s; done
$P -m extras.ai_personas_axis analyze
