#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../.."
for s in ${STAGES:-base sft dpo instruct}; do /opt/arena-env/bin/python -m extras.carryover_balanced score --stage $s; done
/opt/arena-env/bin/python -m extras.carryover_balanced analyze
