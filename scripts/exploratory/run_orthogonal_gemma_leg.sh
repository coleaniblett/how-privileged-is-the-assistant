#!/usr/bin/env bash
# Gemma 4 E4B-it leg of the orthogonalised-steering replication (replaces the Llama leg), then the report. Waits for the Qwen leg's process to end.
# Fits Gemma's formality axis first (the persona-axis fit already exists), regenerates directions.json, runs the 37 settings, writes the report, releases the GPU lock.
set -u
cd "$(dirname "$0")/../.."
P=/opt/arena-env/bin/python
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
while pgrep -f "extras.orthogonal_steering run --model qwen" >/dev/null; do sleep 15; done
echo "qwen leg ended: $(date -u +%T)"
FA_OUT=results/gemma-4-e4b/formality_axis $P -m extras.formality_axis fit --stage google/gemma-4-E4B-it --tokenizer google/gemma-4-E4B-it || exit 1
$P -m extras.orthogonal_steering directions || exit 1
$P -m extras.orthogonal_steering run --model gemma || exit 1
$P -m extras.orthogonal_steering analyze
echo "done: $(date -u +%T)"
