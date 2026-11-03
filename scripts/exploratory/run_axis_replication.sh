#!/usr/bin/env bash
# Replicate the persona-axis separation on an independent role/question set (PA_SET=v2) and test transfer of the original axis.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
export PA_SET=v2
PA_OUT=results/olmo3-7b/persona_axis_v2 bash -c 'for s in base sft dpo instruct; do '$P' -m extras.persona_axis fit --stage $s; done'
$P -m extras.axis_transfer results/olmo3-7b base sft dpo instruct
export T=google/gemma-4-E4B-it
PA_OUT=results/gemma-4-e4b/persona_axis_v2 bash -c 'for m in google/gemma-4-E4B google/gemma-4-E4B-it; do '$P' -m extras.persona_axis fit --stage $m --tokenizer '$T'; done'
$P -m extras.axis_transfer results/gemma-4-e4b gemma-4-E4B gemma-4-E4B-it
