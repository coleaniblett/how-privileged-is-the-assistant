#!/usr/bin/env bash
# Firm up Gemma specificity: 4 more random-direction seeds at the default strengths (+/-1, +/-2) on Gemma 4 E4B instruct.
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
export PA_OUT=results/overgeneralization/early_probes/gemma-4-e4b/persona_axis PA_REPORT=results/overgeneralization/early_probes/gemma-4-e4b/persona_axis_random4.md
for sd in 2 3 4 5; do $P -m extras.persona_axis intervene --stage google/gemma-4-E4B-it --tokenizer google/gemma-4-E4B-it --random $sd; done
$P -m extras.persona_axis analyze
