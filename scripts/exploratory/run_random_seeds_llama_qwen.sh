#!/usr/bin/env bash
# Three more random-direction seeds for the persona-axis steering control on Llama-3.1-8B-Instruct and Qwen3-8B.
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
export PA_OUT=results/other/persona_axis_llama PA_REPORT=results/other/persona_axis_llama_random4.md
for sd in 1 2 3; do $P -m extras.persona_axis intervene --stage meta-llama/Llama-3.1-8B-Instruct --tokenizer meta-llama/Llama-3.1-8B-Instruct --random $sd; done
$P -m extras.persona_axis analyze
export PA_OUT=results/other/persona_axis_qwen PA_REPORT=results/other/persona_axis_qwen_random4.md
for sd in 1 2 3; do $P -m extras.persona_axis intervene --stage Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B --random $sd; done
$P -m extras.persona_axis analyze
