#!/usr/bin/env bash
# After all generation workers are done: judge (Gemma 4 31B instruct, re-downloaded if it was removed), detect, hand-check sheet, analyze (judge results marked provisional).
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1
PY=/opt/arena-env/bin/python
n=8
while [ "$(ls results/overgeneralization/trait_atlas/*/*_replies.jsonl.gz 2>/dev/null | wc -l)" -lt $n ] || pgrep -f "[e]xtras.trait_atlas generate" >/dev/null; do sleep 60; done
$PY -m extras.trait_atlas detect
echo "=== judge $(date -u +%T)"; $PY -m extras.trait_atlas judge --batch-size 16
$PY -m extras.trait_atlas handcheck
echo "=== analyze $(date -u +%T)"; $PY -m extras.trait_atlas analyze
echo "=== finished $(date -u +%T)"
