#!/usr/bin/env bash
# Register-gradient checks on Gemma 4 E4B: other AI/human roles on the persona axis, and helpful-vs-stylized refusal leak. Contamination cannot be checked.
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python; T=google/gemma-4-E4B-it
export PA_OUT=results/overgeneralization/early_probes/gemma-4-e4b/persona_axis AI_OUT=results/overgeneralization/early_probes/gemma-4-e4b/ai_personas_axis AI_REPORT=results/overgeneralization/early_probes/gemma-4-e4b/ai_personas_axis.md
export RG_OUT=results/overgeneralization/early_probes/gemma-4-e4b/role_gradient RG_REPORT=results/overgeneralization/early_probes/gemma-4-e4b/role_gradient.md
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.ai_personas_axis run --stage $m --tokenizer $T; done
$P -m extras.ai_personas_axis analyze
for m in google/gemma-4-E4B google/gemma-4-E4B-it; do $P -m extras.role_gradient run --stage $m --tokenizer $T; done
$P -m extras.role_gradient analyze
