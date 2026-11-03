#!/usr/bin/env bash
# Dominance-matrix pilot (extras/dominance_pilot.py): format calibration on the instruct stage, then one process per
# checkpoint, then the analysis. Results: results/<run>/<NAME>.md and results/<run>/<NAME>/.
#   scripts/run_dominance_pilot.sh                                         # the 5 x 5 pilot (calibrates the format)
#   CONTROL=1 scripts/run_dominance_pilot.sh                               # + the length/structure control
#   DOMINANCE_NAME=dominance_pilot_n15 N=15 CONTROL=1 scripts/run_dominance_pilot.sh   # 15 x 15 rerun (reuses the pilot's format)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/opt/arena-env/bin/python}
export PYTHONPATH=.
export DOMINANCE_NAME=${DOMINANCE_NAME:-dominance_pilot}
N=${N:-5}
RUN=$($PY -c "from core import config; print(config.run_dir())")
OUT="$RUN/$DOMINANCE_NAME"
mkdir -p "$OUT"
[[ -f "data/$DOMINANCE_NAME.jsonl" ]] || $PY -m extras.dominance_pilot build --n "$N"
if [[ "$DOMINANCE_NAME" == dominance_pilot ]]; then
  echo "=== calibrate (instruct) $(date)"
  $PY -m extras.dominance_pilot calibrate --model instruct
else
  cp "$RUN/dominance_pilot/format_choice.json" "$OUT/format_choice.json"  # same format as the pilot
fi
for stage in base sft dpo instruct; do
  echo "=== score $stage $(date)"
  $PY -m extras.dominance_pilot score --model "$stage"
  if [[ "${CONTROL:-0}" == "1" ]]; then
    echo "=== control $stage $(date)"
    $PY -m extras.dominance_pilot score --model "$stage" --control
  fi
done
echo "=== analyze $(date)"
$PY -m extras.dominance_pilot analyze
echo "=== done $(date)"
