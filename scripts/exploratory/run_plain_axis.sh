#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for s in base sft dpo instruct; do $P -m extras.plain_axis run --stage $s --axis-stage instruct; done
$P -m extras.plain_axis analyze
export PL_OUT=results/overgeneralization/early_probes/gemma-4-e4b/plain_axis PL_REPORT=results/overgeneralization/early_probes/gemma-4-e4b/plain_axis.md PA_OUT=results/overgeneralization/early_probes/gemma-4-e4b/persona_axis
T=google/gemma-4-E4B-it
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.plain_axis run --stage $m --tokenizer $T --axis-stage gemma-4-E4B-it; done
$P -m extras.plain_axis analyze
