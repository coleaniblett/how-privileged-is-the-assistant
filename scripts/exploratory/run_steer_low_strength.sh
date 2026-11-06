#!/usr/bin/env bash
# Gemma persona-axis steering at small strengths vs random directions; Llama and Qwen random-direction controls at the default strengths.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
export PA_OUT=results/overgeneralization/early_probes/gemma-4-e4b/persona_axis PA_REPORT=results/overgeneralization/early_probes/gemma-4-e4b/persona_axis_lowstrength.md
G="--stage google/gemma-4-E4B-it --tokenizer google/gemma-4-E4B-it --alphas -0.5 -0.25 0.25 0.5"
$P -m extras.persona_axis intervene $G --tag low
for sd in 0 1; do $P -m extras.persona_axis intervene $G --tag low --random $sd; done
$P -m extras.persona_axis analyze
export PA_OUT=results/overgeneralization/early_probes/other/persona_axis_llama PA_REPORT=results/overgeneralization/early_probes/other/persona_axis_llama_random.md
$P -m extras.persona_axis intervene --stage meta-llama/Llama-3.1-8B-Instruct --tokenizer meta-llama/Llama-3.1-8B-Instruct --random 0
$P -m extras.persona_axis analyze
export PA_OUT=results/overgeneralization/early_probes/other/persona_axis_qwen PA_REPORT=results/overgeneralization/early_probes/other/persona_axis_qwen_random.md
$P -m extras.persona_axis intervene --stage Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B --random 0
$P -m extras.persona_axis analyze
