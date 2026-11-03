#!/usr/bin/env bash
# Start the persona map's OLMo stages on this host's GPU once nobody holds it, hold /tmp/claude-0/gpu.lock while running, release it after. Analysis runs separately once the Qwen stages have been pulled back.
set -u
cd "$(dirname "$0")/../.."
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=persona_map OLMo stages (4 x ~15 min); released when scripts/exploratory/run_persona_map_queued.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
echo "start: $(date -u +%T)"
FAMILIES=olmo SKIP_ANALYZE=1 bash scripts/exploratory/run_persona_map.sh
echo "done: $(date -u +%T)"
