#!/usr/bin/env bash
# Role specification gradient (bare / named / full) on Gemma 4 E4B, Llama-3.1-8B and Qwen3-8B (base vs instruct), each read on its own instruct model's fixed persona axis.
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
run_family () {  # name tokenizer axis-stage pa-dir base instruct
  for form in plain named full; do
    export RR_FORM=$form RR_OUT=results/overgeneralization/early_probes/other/role_register_$1_$form RR_REPORT=results/overgeneralization/early_probes/other/role_register_$1_$form.md PA_OUT=$4
    for m in $5 $6; do $P -m extras.role_register run --stage $m --tokenizer $2 --axis-stage $3; done
    $P -m extras.role_register analyze
  done
}
run_family gemma google/gemma-4-E4B-it gemma-4-E4B-it results/overgeneralization/early_probes/gemma-4-e4b/persona_axis google/gemma-4-E4B google/gemma-4-E4B-it
run_family llama meta-llama/Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct results/overgeneralization/early_probes/other/persona_axis_llama meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct
run_family qwen Qwen/Qwen3-8B Qwen3-8B results/overgeneralization/early_probes/other/persona_axis_qwen Qwen/Qwen3-8B-Base Qwen/Qwen3-8B
