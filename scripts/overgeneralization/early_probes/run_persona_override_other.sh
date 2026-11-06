#!/usr/bin/env bash
# Same override sweep on other families' instruct models (no base checkpoints cached; contamination unchecked).
set -e
cd "$(dirname "$0")/../../.."
export PO_OUT=results/overgeneralization/early_probes/other/persona_override PO_REPORT=results/overgeneralization/early_probes/other/persona_override.md
P=/opt/arena-env/bin/python
$P -m extras.persona_override score --stage meta-llama/Llama-3.1-8B-Instruct --tokenizer meta-llama/Llama-3.1-8B-Instruct
$P -m extras.persona_override score --stage Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B --no-think
$P -m extras.persona_override analyze
