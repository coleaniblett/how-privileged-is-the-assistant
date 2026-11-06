#!/usr/bin/env bash
# False-fact propagation (extras/false_fact.py): bf16 scoring, one process per checkpoint, then the analysis.
# Item selection (GPU, base model, needs network for CounterFact and the infini-gram frequency counts) is separate:
#   python -m extras.false_fact select && python -m extras.false_fact finalize   # data/false_fact.jsonl is committed once reviewed
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/opt/arena-env/bin/python}
export PYTHONPATH=.
[[ -f data/false_fact.jsonl ]] || { echo "data/false_fact.jsonl missing: run select and finalize first"; exit 1; }
mkdir -p "$($PY -c 'from core import config; print(config.measure_dir("false_fact"))')"
for stage in base sft dpo instruct; do
  echo "=== score $stage $(date)"
  $PY -m extras.false_fact score --model "$stage"
done
echo "=== analyze $(date)"
$PY -m extras.false_fact analyze
echo "=== done $(date)"
