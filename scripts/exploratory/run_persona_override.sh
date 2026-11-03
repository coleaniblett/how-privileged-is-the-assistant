#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
for s in ${STAGES:-sft dpo instruct}; do /opt/arena-env/bin/python -m extras.persona_override score --stage $s; done
/opt/arena-env/bin/python -m extras.persona_override analyze
