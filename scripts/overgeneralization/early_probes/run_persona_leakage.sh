#!/usr/bin/env bash
# Persona leakage probes on all OLMo 3 7B stages, then the report.
set -e
cd "$(dirname "$0")/../../.."
for s in ${STAGES:-base sft dpo instruct}; do /opt/arena-env/bin/python -m extras.persona_leakage score --stage $s; done
/opt/arena-env/bin/python -m extras.persona_leakage analyze
