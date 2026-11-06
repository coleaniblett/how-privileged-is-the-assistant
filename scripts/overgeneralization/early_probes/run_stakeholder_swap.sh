#!/usr/bin/env bash
# Stakeholder swap (extras/stakeholder_swap.py): bf16 scoring, one process per checkpoint, then the analysis.
# FP32=1 also runs the fp32 check of one cell (instruct, user reporter, stakeholder A) before the analysis.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/opt/arena-env/bin/python}
export PYTHONPATH=.
for stage in base sft dpo instruct; do
  echo "=== score $stage $(date)"
  $PY -m extras.stakeholder_swap score --model "$stage"
done
if [[ "${FP32:-0}" == "1" ]]; then
  echo "=== fp32check $(date)"
  $PY -m extras.stakeholder_swap fp32check
fi
echo "=== analyze $(date)"
$PY -m extras.stakeholder_swap analyze
echo "=== done $(date)"
