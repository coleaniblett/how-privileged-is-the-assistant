#!/usr/bin/env bash
# Persona-based assistant axis on Gemma 4 E4B (base and instruct); interventions on instruct. Contamination cannot be checked.
set -e
cd "$(dirname "$0")/../.."
export PA_OUT=results/gemma-4-e4b/persona_axis PA_REPORT=results/gemma-4-e4b/persona_axis.md
P=/opt/arena-env/bin/python; T=google/gemma-4-E4B-it
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.persona_axis fit --stage $m --tokenizer $T; done
$P -m extras.persona_axis intervene --stage google/gemma-4-E4B-it --tokenizer $T
$P -m extras.persona_axis analyze
