#!/usr/bin/env bash
# Orthogonalised-steering replication: OLMo 3 7B instruct, Qwen3-8B, Gemma 4 E4B-it (Llama was swapped out to stay on the repo's model list). Resumable (one jsonl per setting under results/overgeneralization/orthogonal_steering/gens/<model>/).
# Needs: persona-axis fits (results/overgeneralization/early_probes/olmo3-7b/persona_axis, results/overgeneralization/early_probes/other/persona_axis_qwen, results/overgeneralization/early_probes/gemma-4-e4b/persona_axis) and formality fits (results/overgeneralization/early_probes/olmo3-7b/formality_axis, results/overgeneralization/early_probes/other/formality_axis_qwen, results/overgeneralization/early_probes/gemma-4-e4b/formality_axis).
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
$P -m extras.orthogonal_steering directions
for m in ${MODELS:-olmo qwen gemma}; do $P -m extras.orthogonal_steering run --model $m; done
