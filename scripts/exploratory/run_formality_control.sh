#!/usr/bin/env bash
# Register confound control on OLMo: fit a formality axis per stage, compare with the persona axis, then steer instruct along the formality axis (same protocol as the persona axis).
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for s in base sft dpo instruct; do $P -m extras.formality_axis fit --stage $s; done
$P -m extras.formality_axis compare
PA_OUT=results/olmo3-7b/formality_axis PA_REPORT=results/olmo3-7b/formality_steering.md $P -m extras.persona_axis intervene --stage instruct
PA_OUT=results/olmo3-7b/formality_axis PA_REPORT=results/olmo3-7b/formality_steering.md $P -m extras.persona_axis analyze
