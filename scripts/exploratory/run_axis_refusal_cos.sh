#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for s in base sft dpo instruct; do $P -m extras.axis_refusal_cos run --stage $s; done
$P -m extras.axis_refusal_cos analyze
export RC_OUT=results/gemma-4-e4b/axis_refusal_cos
T=google/gemma-4-E4B-it
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.axis_refusal_cos run --stage $m --tokenizer $T --pa-out results/gemma-4-e4b/persona_axis; done
$P -m extras.axis_refusal_cos analyze
