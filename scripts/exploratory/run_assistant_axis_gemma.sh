#!/usr/bin/env bash
# Slot x content assistant-direction map on Gemma 4 E4B (base, then instruct). Contamination of post-training data cannot be checked.
set -e
cd "$(dirname "$0")/../.."
export AX_OUT=results/overgeneralization/early_probes/gemma-4-e4b/assistant_axis AX_REPORT=results/overgeneralization/early_probes/gemma-4-e4b/assistant_axis.md
P=/opt/arena-env/bin/python
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.assistant_axis score --stage $m --tokenizer google/gemma-4-E4B-it; done
$P -m extras.assistant_axis analyze
