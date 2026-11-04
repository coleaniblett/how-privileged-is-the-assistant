#!/usr/bin/env bash
# Specificity control for persona-axis steering: norm-matched random directions (2 seeds), OLMo instruct and Gemma instruct.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for sd in 0 1; do $P -m extras.persona_axis intervene --stage instruct --random $sd; done
PA_REPORT=results/olmo3-7b/persona_axis_random.md $P -m extras.persona_axis analyze
export PA_OUT=results/gemma-4-e4b/persona_axis PA_REPORT=results/gemma-4-e4b/persona_axis_random.md
for sd in 0 1; do $P -m extras.persona_axis intervene --stage google/gemma-4-E4B-it --tokenizer google/gemma-4-E4B-it --random $sd; done
$P -m extras.persona_axis analyze
