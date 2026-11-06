#!/usr/bin/env bash
# Introspection across personas in the Macar et al. paradigm (extras/introspection_macar.py), Qwen3-8B (thinking off) in bf16 by default (MODEL=gemma-4-e4b for the fallback).
# vectors -> Yes-No logit-gap sweep -> full-generation confirmation of the top cells -> operating point -> Step 1 (gate) -> Step 2 (only if the gate passes).
# The judge validation (blindsample -> an independent labeller -> validate) is a manual step between "judge" and the final "analyze".
# One model at a time on the 24 GB card; take /tmp/claude-0/gpu.lock first. Exit code 0 also when the gate fails (the report says so).
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/opt/arena-env/bin/python}
export PYTHONPATH=.
M="--model ${MODEL:-qwen3-8b} --precision ${PRECISION:-bf16}"
$PY -m extras.introspection_macar vectors $M
$PY -m extras.introspection_macar sweep $M
$PY -m extras.introspection_macar generate --phase confirm $M
$PY -m extras.introspection_macar judge --phase confirm $M
$PY -m extras.introspection_macar choose $M
$PY -m extras.introspection_macar generate --phase step1 $M
$PY -m extras.introspection_macar judge --phase step1 $M
set +e
$PY -m extras.introspection_macar analyze --phase step1 $M
rc=$?
set -e
if [[ $rc -eq 3 ]]; then echo "=== gate FAILED: Step 2 not run"; exit 0; fi
[[ $rc -eq 0 ]] || exit $rc
$PY -m extras.introspection_macar generate --phase step2 $M
$PY -m extras.introspection_macar judge --phase step2 $M
$PY -m extras.introspection_macar analyze --phase step2 $M
echo "=== done"
