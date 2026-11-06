#!/usr/bin/env bash
# Steer OLMo instruct along the persona axis with its formality component removed (same strengths, same gap-norm unit).
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
$P -m extras.persona_axis intervene --stage instruct --orth results/overgeneralization/early_probes/olmo3-7b/formality_axis
PA_REPORT=results/overgeneralization/early_probes/olmo3-7b/persona_axis_orth.md $P -m extras.persona_axis analyze
