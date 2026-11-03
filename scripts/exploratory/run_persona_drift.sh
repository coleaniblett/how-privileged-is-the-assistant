#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for s in ${STAGES:-base sft dpo instruct}; do $P -m extras.persona_drift run --stage $s; done
$P -m extras.persona_drift analyze
