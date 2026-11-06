#!/usr/bin/env bash
# Helpful-vs-stylized 'never refuses' refusal check on Llama-3.1-8B and Qwen3-8B (base and instruct).
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
export RG_OUT=results/overgeneralization/early_probes/other/role_gradient_llama RG_REPORT=results/overgeneralization/early_probes/other/role_gradient_llama.md
T=meta-llama/Llama-3.1-8B-Instruct
for m in meta-llama/Llama-3.1-8B $T; do $P -m extras.role_gradient run --stage $m --tokenizer $T; done
$P -m extras.role_gradient analyze
export RG_OUT=results/overgeneralization/early_probes/other/role_gradient_qwen RG_REPORT=results/overgeneralization/early_probes/other/role_gradient_qwen.md
T=Qwen/Qwen3-8B
for m in Qwen/Qwen3-8B-Base $T; do $P -m extras.role_gradient run --stage $m --tokenizer $T; done
$P -m extras.role_gradient analyze
