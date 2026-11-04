#!/usr/bin/env bash
# Chain (2026-11-04): after run_mechanism_checks.sh, redo OLMo base H2/H3 (they hit a CUDA OOM at batch 48 before the batch-16 fix), re-analyze, then user-slot scoring.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
PY=/opt/arena-env/bin/python
while pgrep -f "[r]un_mechanism_checks.sh" >/dev/null; do sleep 20; done
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec H redo olmo base h2/h3; released when this script ends" > /tmp/claude-0/gpu.lock
for p in h2 h3; do $PY -m extras.mechanism_checks $p --family olmo3-7b --stage base || echo "FAILED $p olmo base"; done
$PY -m extras.mechanism_checks analyze
rm -f /tmp/claude-0/gpu.lock
exec bash scripts/exploratory/run_reply_in_user.sh
