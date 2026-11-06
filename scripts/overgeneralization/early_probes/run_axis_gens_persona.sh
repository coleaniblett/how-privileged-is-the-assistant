#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../.."
export AG_AXIS=persona
P=/opt/arena-env/bin/python
for s in ${STAGES:-base sft dpo instruct}; do $P -m extras.axis_gens score --stage $s; done
$P -m extras.axis_gens analyze
