#!/usr/bin/env bash
# Leakage matrix (register x refusal-under-instruction) across contexts: OLMo 3 7B base/SFT/DPO/instruct, Qwen3-8B-Base/Qwen3-8B, Gemma 4 E4B/E4B-it.
# Estimated ~12 min per stage (8 stages) on the RTX 4090. Resumable per stage. Needs the persona-axis fits (results/overgeneralization/early_probes/olmo3-7b/persona_axis, results/overgeneralization/early_probes/other/persona_axis_qwen, results/overgeneralization/early_probes/gemma-4-e4b/persona_axis).
# Do not start while another GPU job is running (24 GB). Report: results/overgeneralization/dominance_matrix/dominance_matrix.md
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for f in ${FAMILIES:-olmo qwen gemma}; do $P -m extras.leakage_matrix run --family $f; done
$P -m extras.leakage_matrix analyze
