#!/usr/bin/env bash
# Steering interventions along the persona axis on Llama-3.1-8B-Instruct and Qwen3-8B (thinking off).
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
export PA_OUT=results/other/persona_axis_llama PA_REPORT=results/other/persona_axis_llama.md
$P -m extras.persona_axis intervene --stage meta-llama/Llama-3.1-8B-Instruct --tokenizer meta-llama/Llama-3.1-8B-Instruct
$P -m extras.persona_axis analyze
export PA_OUT=results/other/persona_axis_qwen PA_REPORT=results/other/persona_axis_qwen.md
$P -m extras.persona_axis intervene --stage Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B
$P -m extras.persona_axis analyze
