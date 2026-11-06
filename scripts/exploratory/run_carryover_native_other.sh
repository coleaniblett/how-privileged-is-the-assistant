#!/usr/bin/env bash
# Native-slot carryover on other families' instruct models (no contamination check; no base).
set -e
cd "$(dirname "$0")/../.."
export CB_HOLDS=${HOLDS:-nat_assistant,nat_user,char} CB_FACTS=1
P=/opt/arena-env/bin/python
for m in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B; do
  tag=$(basename $m)
  CB_OUT=results/overgeneralization/early_probes/other/${PREFIX:-carryover_native}_$tag CB_TOKENIZER=$m $P -m extras.carryover_balanced score --stage $m
  CB_OUT=results/overgeneralization/early_probes/other/${PREFIX:-carryover_native}_$tag CB_TOKENIZER=$m $P -m extras.carryover_balanced analyze
done
