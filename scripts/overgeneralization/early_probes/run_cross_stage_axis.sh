#!/usr/bin/env bash
# Cross-stage axis transfer: steer SFT with the instruct axis and instruct with the SFT axis (OLMo).
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
$P -m extras.persona_axis intervene --stage sft --axis-from instruct
$P -m extras.persona_axis intervene --stage instruct --axis-from sft
$P -m extras.persona_axis intervene --stage dpo --axis-from instruct
PA_REPORT=results/overgeneralization/early_probes/olmo3-7b/persona_axis_cross.md $P -m extras.persona_axis analyze
