#!/usr/bin/env bash
# Leakage matrix (register x refusal-under-instruction) across contexts: OLMo 3 7B base/SFT/DPO/instruct, Qwen3-8B-Base/Qwen3-8B, Llama-3.1-8B/-Instruct.
# Estimated ~12 min per stage (8 stages) on the RTX 4090. Resumable per stage. Needs the persona-axis fits (results/olmo3-7b/persona_axis, results/other/persona_axis_{qwen,llama}).
# Do not start while another GPU job is running (24 GB). Report: results/dominance_matrix/dominance_matrix.md
set -e
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
for f in ${FAMILIES:-olmo qwen llama}; do $P -m extras.leakage_matrix run --family $f; done
$P -m extras.leakage_matrix analyze
