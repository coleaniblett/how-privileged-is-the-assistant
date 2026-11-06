#!/usr/bin/env bash
# Persona-based axis fit (no interventions) on Llama-3.1-8B and Qwen3-8B base vs instruct. Contamination cannot be checked; Qwen3-8B is the hybrid-thinking post-trained model, run with thinking off.
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
export PA_OUT=results/overgeneralization/early_probes/other/persona_axis_llama PA_REPORT=results/overgeneralization/early_probes/other/persona_axis_llama.md
T=meta-llama/Llama-3.1-8B-Instruct
for m in meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct; do $P -m extras.persona_axis fit --stage $m --tokenizer $T; done
$P -m extras.persona_axis analyze
export PA_OUT=results/overgeneralization/early_probes/other/persona_axis_qwen PA_REPORT=results/overgeneralization/early_probes/other/persona_axis_qwen.md
T=Qwen/Qwen3-8B
for m in Qwen/Qwen3-8B-Base Qwen/Qwen3-8B; do $P -m extras.persona_axis fit --stage $m --tokenizer $T; done
$P -m extras.persona_axis analyze
