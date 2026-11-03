#!/usr/bin/env bash
# Introspection across personas (extras/introspection_personas.py), Llama-3.1-8B-Instruct by default (MODEL=olmo for OLMo 3 7B Instruct).
# Sweep -> operating point -> Step 1 (gate) -> Step 2 (only if the gate passes). Each stage loads one model at a time (generation model,
# then the judge), so it fits the 24 GB card. Exit code 0 also when the gate fails (the report says so).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/opt/arena-env/bin/python}
export PYTHONPATH=.
M="--model ${MODEL:-llama}"
$PY -m extras.introspection_personas vectors $M
$PY -m extras.introspection_personas generate --phase sweep $M
$PY -m extras.introspection_personas judge --phase sweep $M
$PY -m extras.introspection_personas choose $M
$PY -m extras.introspection_personas generate --phase step1 $M
$PY -m extras.introspection_personas judge --phase step1 $M
set +e
$PY -m extras.introspection_personas analyze --phase step1 $M
rc=$?
set -e
if [[ $rc -eq 3 ]]; then echo "=== gate FAILED: Step 2 not run"; exit 0; fi
[[ $rc -eq 0 ]] || exit $rc
$PY -m extras.introspection_personas generate --phase step2 $M
$PY -m extras.introspection_personas judge --phase step2 $M
$PY -m extras.introspection_personas analyze --phase step2 $M
echo "=== done"
