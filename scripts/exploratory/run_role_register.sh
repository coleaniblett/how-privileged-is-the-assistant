#!/usr/bin/env bash
# OLMo 4 stages and Gemma base/instruct, all read on the post-trained model's fixed persona axis.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for s in base sft dpo instruct; do $P -m extras.role_register run --stage $s --axis-stage instruct; done
$P -m extras.role_register analyze
export RR_OUT=results/gemma-4-e4b/role_register RR_REPORT=results/gemma-4-e4b/role_register.md PA_OUT=results/gemma-4-e4b/persona_axis
T=google/gemma-4-E4B-it
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.role_register run --stage $m --tokenizer $T --axis-stage gemma-4-E4B-it; done
$P -m extras.role_register analyze
