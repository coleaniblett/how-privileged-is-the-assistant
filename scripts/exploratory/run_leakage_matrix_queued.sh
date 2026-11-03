#!/usr/bin/env bash
# Start the leakage matrix on this host's GPU once nobody holds it (waits for /tmp/claude-0/gpu.lock to disappear and the card to be free), hold the lock while running, release it after.
# Then the OLMo sensitivity analysis on the independent v2 axis. Models are all in the local HF cache; nothing is downloaded.
set -u
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=leakage_matrix (8 stages, ~1.6 h); released when scripts/exploratory/run_leakage_matrix_queued.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
echo "start: $(date -u +%T)"
bash scripts/exploratory/run_leakage_matrix.sh || exit 1
$P -m extras.leakage_matrix analyze --coord coord_v2 || true
echo "done: $(date -u +%T)"
