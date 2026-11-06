#!/usr/bin/env bash
# Persona map (extras/persona_map.py): 180 personas x 40 probes + the default Assistant, activations at the last reply token, six analyses -> results/overgeneralization/persona_map/persona_map.md.
# FAMILIES selects what to generate (default: both; resumable per stage). On this host the OLMo stages use the local GPU; the Qwen3-8B stages can run on arena9-nurse (see remote.sh), then
# `remote.sh pull-results results/overgeneralization/persona_map/stages/qwen`. The analysis needs every stage's files under results/overgeneralization/persona_map/stages/ on this host.
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
[ -f data/persona_map_personas.json ] || $P -m extras.persona_map design
for f in ${FAMILIES:-olmo qwen}; do $P -m extras.persona_map run --family $f; done
[ -n "${SKIP_ANALYZE:-}" ] || $P -m extras.persona_map analyze
