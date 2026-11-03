#!/usr/bin/env bash
# Implant-and-spread (spec D) for FAMILY (default olmo3-7b): wait for the GPU, hold /tmp/claude-0/gpu.lock, generate replies, run the 24 LoRA runs, release, analyze.
set -u
cd "$(dirname "$0")/../.."
FAMILY="${FAMILY:-olmo3-7b}"
PY=/opt/arena-env/bin/python
OUT="results/implant_spread/$FAMILY"
mkdir -p /tmp/claude-0
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=implant_spread $FAMILY (24 LoRA runs, ~3 h); released when scripts/exploratory/run_implant_spread.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
echo "start: $(date -u +%T)"
$PY -m extras.implant_spread gen-replies --family "$FAMILY" --out "$OUT" || exit 1
$PY -m extras.implant_spread run --family "$FAMILY" --stages base,instruct --out "$OUT" || exit 1
$PY -m extras.implant_spread analyze --family "$FAMILY" --out "$OUT"
echo "done: $(date -u +%T)"
