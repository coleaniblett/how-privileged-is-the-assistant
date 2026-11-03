#!/usr/bin/env bash
# Orthogonalised-steering replication: OLMo 3 7B instruct, Qwen3-8B, Llama-3.1-8B-Instruct. Resumable (one jsonl per setting under results/orthogonal_steering/gens/<model>/).
# Needs: persona-axis fits (results/olmo3-7b/persona_axis, results/other/persona_axis_{qwen,llama}) and formality fits (results/olmo3-7b/formality_axis, results/other/formality_axis_{qwen,llama}).
set -e
cd "$(dirname "$0")/.."
P=/opt/arena-env/bin/python
$P -m extras.orthogonal_steering directions
for m in ${MODELS:-olmo qwen llama}; do $P -m extras.orthogonal_steering run --model $m; done
