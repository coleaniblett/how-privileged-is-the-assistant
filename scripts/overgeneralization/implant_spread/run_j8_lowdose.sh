#!/usr/bin/env bash
# Spec J8 (plan: docs/decisions.md 2026-11-05): lower-dose implant-and-spread, OLMo base + instruct. Log /tmp/claude-0/j8.log
set -uo pipefail
cd "$(dirname "$0")/../../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
OUT=results/overgeneralization/implant_spread/olmo3-7b_lowdose
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=J8 low-dose implant; released when run_j8_lowdose.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
mkdir -p $OUT
[ -f $OUT/replies.jsonl ] || cp /root/how-privileged-is-the-assistant/results/overgeneralization/implant_spread/olmo3-7b/replies.jsonl $OUT/replies.jsonl
echo "start $(date -u +%T)"
$PY -m extras.implant_spread run --family olmo3-7b --stages base,instruct --out $OUT --epochs 1 --max-steps 8 || echo FAILED
$PY -m extras.implant_spread analyze --family olmo3-7b --out $OUT
echo "done $(date -u +%T)"
